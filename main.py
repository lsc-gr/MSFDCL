import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc as sk_auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from hypergraph_utils import *
from hypergraph_utils import knn_local_enhance
from model import Model
from glhmf import GLHMF
from hgcl import hgcl_loss, channel_hgcl_loss


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def g_to_edge_index(G_tensor, threshold=0.1):
    G_np = G_tensor.detach().cpu().numpy()
    return torch.tensor(
        np.array(np.where(G_np > threshold)),
        dtype=torch.long,
    )


def align_feat(feat, target_dim=50):
    feat = np.asarray(feat, dtype=np.float64)
    if feat.shape[1] < target_dim:
        pad = np.zeros(
            (feat.shape[0], target_dim - feat.shape[1]),
            dtype=np.float64,
        )
        return np.concatenate([feat, pad], axis=1)
    return feat[:, :target_dim]



def gip_row_similarity(A):
    A = np.asarray(A, dtype=np.float64)
    sq_norm_sum = np.sum(np.square(np.linalg.norm(A, axis=1)))
    gamma = A.shape[0] / max(sq_norm_sum, 1e-12)

    diff = A[:, None, :] - A[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * dist2)


def gip_col_similarity(A):
    A = np.asarray(A, dtype=np.float64)
    AT = A.T
    sq_norm_sum = np.sum(np.square(np.linalg.norm(AT, axis=1)))
    gamma = AT.shape[0] / max(sq_norm_sum, 1e-12)

    diff = AT[:, None, :] - AT[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * dist2)


def integrate_similarity(external_sim, gip_sim, w=0.3):
    external_sim = np.asarray(external_sim, dtype=np.float64)
    gip_sim = np.asarray(gip_sim, dtype=np.float64)

    if external_sim.shape != gip_sim.shape:
        raise ValueError(
            f"Similarity shape mismatch: external={external_sim.shape}, "
            f"GIP={gip_sim.shape}"
        )

    sim = w * external_sim + (1.0 - w) * gip_sim
    sim = (sim + sim.T) / 2.0
    return sim



def build_balanced_samples(A, seed=42):
    A = np.asarray(A)
    pos_pairs = np.argwhere(A != 0)
    neg_pairs = np.argwhere(A == 0)

    if len(neg_pairs) < len(pos_pairs):
        raise RuntimeError(
            f"Not enough zero entries for 1:1 sampling: "
            f"positive={len(pos_pairs)}, zero={len(neg_pairs)}"
        )

    rng = np.random.RandomState(seed)
    neg_ids = rng.choice(len(neg_pairs), size=len(pos_pairs), replace=False)
    sampled_neg_pairs = neg_pairs[neg_ids]

    pos_samples = np.column_stack([
        pos_pairs,
        np.ones(len(pos_pairs), dtype=np.int64),
    ])
    neg_samples = np.column_stack([
        sampled_neg_pairs,
        np.zeros(len(sampled_neg_pairs), dtype=np.int64),
    ])

    samples = np.concatenate([pos_samples, neg_samples], axis=0).astype(np.int64)
    return samples


def build_kfold_indices(samples, cv_num=5, seed=42):
    kf = KFold(n_splits=cv_num, shuffle=True, random_state=seed)
    return list(kf.split(samples))


def build_pair_mask(shape, samples):
    mask = np.zeros(shape, dtype=bool)
    rows = samples[:, 0].astype(np.int64)
    cols = samples[:, 1].astype(np.int64)
    mask[rows, cols] = True
    return mask



def build_static_similarity_hypergraphs(MM, DD, dtype=torch.float64,
                                        mir_knn=50, drug_knn=30):
    HHMM = construct_H_with_KNN(
        knn_local_enhance(MM, k=min(mir_knn, max(1, MM.shape[0] - 1)), thresh=0.0)
    )
    HMM = generate_G_from_H(HHMM).to(dtype=dtype)

    HHDD = construct_H_with_KNN(
        knn_local_enhance(DD, k=min(drug_knn, max(1, DD.shape[0] - 1)), thresh=0.0)
    )
    HDD = generate_G_from_H(HHDD).to(dtype=dtype)

    return HHMM, HMM, HHDD, HDD


def build_global_objects(
    A_full,
    MM,
    DD,
    linear_rank=50,
    glhmf_alpha=0.8,
    glhmf_k=10,
    glhmf_lambda=0.0001,
    glhmf_max_iter=1000,
):
    print("[GL-HMF] Reconstruction using FULL association matrix...")

    glhmf = GLHMF(
        rank=linear_rank,
        lamb=glhmf_lambda,
        max_iter=glhmf_max_iter,
        alpha=glhmf_alpha,
        k=glhmf_k,
    )

    M_global, M_local, A_mir_g, B_drug_g, A_mir_l, B_drug_l = \
        glhmf.fit_transform(A_full, MM, DD)

    print(
        "[GL-HMF] Done. "
        f"M_global=[{M_global.min():.4f}, {M_global.max():.4f}], "
        f"M_local=[{M_local.min():.4f}, {M_local.max():.4f}]"
    )

    HHMD_G = construct_H_with_KNN(M_global)
    HMD_G = generate_G_from_H(HHMD_G).double()

    HHMD_L = construct_H_with_KNN(M_local)
    HMD_L = generate_G_from_H(HHMD_L).double()

    HHDM_G = construct_H_with_KNN(M_global.T)
    HDM_G = generate_G_from_H(HHDM_G).double()

    HHDM_L = construct_H_with_KNN(M_local.T)
    HDM_L = generate_G_from_H(HHDM_L).double()

    edge_index_md_g = g_to_edge_index(HMD_G, threshold=0.1)
    edge_index_md_l = g_to_edge_index(HMD_L, threshold=0.1)
    edge_index_dm_g = g_to_edge_index(HDM_G, threshold=0.1)
    edge_index_dm_l = g_to_edge_index(HDM_L, threshold=0.1)

    mir_lin_g = torch.from_numpy(align_feat(A_mir_g, linear_rank)).double()
    mir_lin_l = torch.from_numpy(align_feat(A_mir_l, linear_rank)).double()
    drug_lin_g = torch.from_numpy(align_feat(B_drug_g, linear_rank)).double()
    drug_lin_l = torch.from_numpy(align_feat(B_drug_l, linear_rank)).double()

    return {
        "M_global": M_global,
        "M_local": M_local,
        "HHMD_G": HHMD_G,
        "HHMD_L": HHMD_L,
        "HHDM_G": HHDM_G,
        "HHDM_L": HHDM_L,
        "HMD_G": HMD_G,
        "HMD_L": HMD_L,
        "HDM_G": HDM_G,
        "HDM_L": HDM_L,
        "edge_index_md_g": edge_index_md_g,
        "edge_index_md_l": edge_index_md_l,
        "edge_index_dm_g": edge_index_dm_g,
        "edge_index_dm_l": edge_index_dm_l,
        "mir_lin_g": mir_lin_g,
        "mir_lin_l": mir_lin_l,
        # Keep dis_* aliases because the current Model interface uses them.
        "dis_lin_g": drug_lin_g,
        "dis_lin_l": drug_lin_l,
    }


def move_objects_to_device(obj, device):
    keys = [
        "HMD_G", "HMD_L", "HDM_G", "HDM_L",
        "edge_index_md_g", "edge_index_md_l",
        "edge_index_dm_g", "edge_index_dm_l",
        "mir_lin_g", "mir_lin_l", "dis_lin_g", "dis_lin_l",
    ]
    for key in keys:
        obj[key] = obj[key].to(device)
    return obj

def train_one_epoch(
    model,
    optimizer,
    pos_weight,
    lambdas,
    A,
    AT,
    train_pair_mask_tensor,
    mir_feat,
    drug_feat,
    HMM,
    HDD,
    obj,
):
    model.train()

    (
        reconstructionMM,
        reconstructionMD_G,
        reconstructionMD_L,
        result,
        recover,
        mir_view_mm,
        mir_view_md_g,
        mir_view_md_l,
        _,
        drug_view_dd,
        drug_view_dm_l,
        drug_view_dm_g,
        _,
        mir_final,
        drug_final,
        _,
        w_mir,
        w_drug,
        mir_g_dis,
        mir_l_dis,
        drug_g_dis,
        drug_l_dis,
    ) = model(
        AT,
        A,
        mir_feat,
        drug_feat,
        HMM,
        HDD,
        obj["HMD_G"],
        obj["HMD_L"],
        obj["HDM_G"],
        obj["HDM_L"],
        obj["edge_index_md_g"],
        obj["edge_index_md_l"],
        obj["edge_index_dm_g"],
        obj["edge_index_dm_l"],
        obj["mir_lin_g"],
        obj["mir_lin_l"],
        obj["dis_lin_g"],
        obj["dis_lin_l"],
    )

    labels = torch.masked_select(A, train_pair_mask_tensor)

    reMM = torch.masked_select(reconstructionMM.t(), train_pair_mask_tensor)
    reMD_G = torch.masked_select(reconstructionMD_G.t(), train_pair_mask_tensor)
    reMD_L = torch.masked_select(reconstructionMD_L.t(), train_pair_mask_tensor)
    rec = torch.masked_select(recover.t(), train_pair_mask_tensor)

    loss_main = (
        F.binary_cross_entropy_with_logits(reMM, labels, pos_weight=pos_weight)
        + F.binary_cross_entropy_with_logits(reMD_G, labels, pos_weight=pos_weight)
        + F.binary_cross_entropy_with_logits(reMD_L, labels, pos_weight=pos_weight)
        + F.binary_cross_entropy_with_logits(rec, labels, pos_weight=pos_weight)
    )

    loss_hgcl = torch.tensor(0.0, dtype=A.dtype, device=A.device)
    if mir_view_md_l is not None and drug_view_dm_g is not None:
        loss_hgcl = (
            hgcl_loss(
                mir_view_md_g,
                mir_view_md_l,
                obj["HHMD_G"],
                tau=0.1,
            )
            + hgcl_loss(
                drug_view_dm_g,
                drug_view_dm_l,
                obj["HHDM_G"],
                tau=0.1,
            )
        )

    loss_disen = torch.tensor(0.0, dtype=A.dtype, device=A.device)

    if mir_g_dis is not None and mir_l_dis is not None:
        loss_disen = loss_disen + channel_hgcl_loss(
            mir_g_dis,
            mir_l_dis,
            obj["HHMD_G"],
            k=2,
            tau=0.1,
            ortho_weight=0.05,
        )

    if drug_g_dis is not None and drug_l_dis is not None:
        loss_disen = loss_disen + channel_hgcl_loss(
            drug_g_dis,
            drug_l_dis,
            obj["HHDM_G"],
            k=2,
            tau=0.1,
            ortho_weight=0.05,
        )

    loss = (
        lambdas["main"] * loss_main
        + lambdas["hgcl"] * loss_hgcl
        + lambdas["disen"] * loss_disen
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "loss_main": float(loss_main.detach().cpu().item()),
        "loss_hgcl": float(loss_hgcl.detach().cpu().item()),
        "loss_disen": float(loss_disen.detach().cpu().item()),
    }


@torch.no_grad()
def infer_score(
    model,
    A,
    AT,
    mir_feat,
    drug_feat,
    HMM,
    HDD,
    obj,
):
    model.eval()

    outputs = model(
        AT,
        A,
        mir_feat,
        drug_feat,
        HMM,
        HDD,
        obj["HMD_G"],
        obj["HMD_L"],
        obj["HDM_G"],
        obj["HDM_L"],
        obj["edge_index_md_g"],
        obj["edge_index_md_l"],
        obj["edge_index_dm_g"],
        obj["edge_index_dm_l"],
        obj["mir_lin_g"],
        obj["mir_lin_l"],
        obj["dis_lin_g"],
        obj["dis_lin_l"],
    )

    recover = outputs[4]
    return recover.t().detach().cpu().numpy()



def compute_binary_metrics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    roc_auc = roc_auc_score(y_true, y_score)

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_true, y_score
    )
    pr_auc = sk_auc(pr_recall, pr_precision)

    if len(pr_thresholds) == 0:
        best_threshold = 0.5
    else:
        p = pr_precision[:-1]
        r = pr_recall[:-1]
        f1_values = 2 * p * r / (p + r + 1e-12)
        best_idx = int(np.nanargmax(f1_values))
        best_threshold = float(pr_thresholds[best_idx])

    y_pred = (y_score >= best_threshold).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": float(roc_auc),
        "AUPR": float(pr_auc),
        "Accuracy": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
        "MCC": float(mcc),
        "Threshold": float(best_threshold),
    }


def evaluate_samples(score_matrix, test_samples):
    rows = test_samples[:, 0].astype(np.int64)
    cols = test_samples[:, 1].astype(np.int64)
    y_true = test_samples[:, 2].astype(np.int64)
    y_score = score_matrix[rows, cols]

    metrics = compute_binary_metrics(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)

    return metrics, {
        "y_true": y_true,
        "y_score": y_score,
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
    }

def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    )
    print(f"[Device] {device}")

    os.makedirs(args.result_dir, exist_ok=True)

    A_np = np.loadtxt(
        os.path.join(args.data_dir, "guanlianmatrix.txt")
    ).astype(np.float64)
    MM_external = np.loadtxt(
        os.path.join(args.data_dir, "m-mmatrix.txt")
    ).astype(np.float64)
    DD_external = np.loadtxt(
        os.path.join(args.data_dir, "d-d2matrix.txt")
    ).astype(np.float64)

    num_mir, num_drug = A_np.shape
    known_num = int(np.sum(A_np != 0))

    if MM_external.shape != (num_mir, num_mir):
        raise ValueError(
            f"m-mmatrix shape {MM_external.shape} does not match "
            f"miRNA count {num_mir}."
        )
    if DD_external.shape != (num_drug, num_drug):
        raise ValueError(
            f"d-d2matrix shape {DD_external.shape} does not match "
            f"drug count {num_drug}."
        )

    print("=" * 72)
    print(f"[Data] Association matrix: {A_np.shape} (miRNA x drug)")
    print(f"[Data] Known positive associations: {known_num}")
    print(f"[Data] miRNA similarity: {MM_external.shape}")
    print(f"[Data] drug similarity: {DD_external.shape}")
    print("=" * 72)


    GIP_MM = gip_row_similarity(A_np)
    GIP_DD = gip_col_similarity(A_np)

    MM = integrate_similarity(MM_external, GIP_MM, w=args.sim_weight)
    DD = integrate_similarity(DD_external, GIP_DD, w=args.sim_weight)

    np.savetxt(os.path.join(args.result_dir, "integrated_miRNA_similarity.txt"), MM)
    np.savetxt(os.path.join(args.result_dir, "integrated_drug_similarity.txt"), DD)

    print(
        f"[Similarity] S = {args.sim_weight:.3f} * external + "
        f"{1.0 - args.sim_weight:.3f} * GIP"
    )

    samples = build_balanced_samples(A_np, seed=args.seed)
    cv_indices = build_kfold_indices(
        samples,
        cv_num=args.cv_num,
        seed=args.seed,
    )

    np.save(
        os.path.join(args.result_dir, f"balanced_samples_seed{args.seed}.npy"),
        samples,
    )

    n_pos = int(np.sum(samples[:, 2] == 1))
    n_neg = int(np.sum(samples[:, 2] == 0))
    print(f"[Samples] positive={n_pos}, negative={n_neg}, total={len(samples)}")

    split_save = {}
    for fold_id, (train_ids, test_ids) in enumerate(cv_indices, start=1):
        split_save[f"fold{fold_id}_train_ids"] = train_ids
        split_save[f"fold{fold_id}_test_ids"] = test_ids
    np.savez(
        os.path.join(args.result_dir, f"vcmds_kfold_seed{args.seed}.npz"),
        **split_save,
    )

    _, HMM, _, HDD = build_static_similarity_hypergraphs(
        MM,
        DD,
        mir_knn=args.mir_similarity_knn,
        drug_knn=args.drug_similarity_knn,
    )
    HMM = HMM.to(device)
    HDD = HDD.to(device)

    obj = build_global_objects(
        A_full=A_np,
        MM=MM,
        DD=DD,
        linear_rank=args.linear_rank,
        glhmf_alpha=args.glhmf_alpha,
        glhmf_k=args.glhmf_k,
        glhmf_lambda=args.glhmf_lambda,
        glhmf_max_iter=args.glhmf_max_iter,
    )
    obj = move_objects_to_device(obj, device)

    A = torch.from_numpy(A_np).to(device=device, dtype=torch.float64)
    AT = A.t().contiguous()

    mir_feat = torch.eye(num_mir, dtype=torch.float64, device=device)
    drug_feat = torch.eye(num_drug, dtype=torch.float64, device=device)

    lambdas = {
        "main": args.lambda_main,
        "hgcl": args.lambda_hgcl,
        "disen": args.lambda_disen,
    }

    fold_rows = []

    for fold_id, (train_ids, test_ids) in enumerate(cv_indices, start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold_id}/{args.cv_num}")
        print("=" * 80)

        set_seed(args.seed + fold_id)

        train_samples = samples[train_ids]
        test_samples = samples[test_ids]

        tr_pos = int(np.sum(train_samples[:, 2] == 1))
        tr_neg = int(np.sum(train_samples[:, 2] == 0))
        te_pos = int(np.sum(test_samples[:, 2] == 1))
        te_neg = int(np.sum(test_samples[:, 2] == 0))

        print(
            f"[Fold {fold_id}] train: pos={tr_pos}, neg={tr_neg}, total={len(train_samples)}"
        )
        print(
            f"[Fold {fold_id}] test : pos={te_pos}, neg={te_neg}, total={len(test_samples)}"
        )

        train_pair_mask = build_pair_mask(A_np.shape, train_samples)
        train_pair_mask_tensor = torch.from_numpy(train_pair_mask).to(device=device)

        pos_weight_value = tr_neg / max(tr_pos, 1)
        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float64,
            device=device,
        )

        model = Model(
            num_in_node=num_drug,
            num_in_edge=num_mir,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            linear_rank=args.linear_rank,
        ).double().to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(args.epochs):
            losses = train_one_epoch(
                model=model,
                optimizer=optimizer,
                pos_weight=pos_weight,
                lambdas=lambdas,
                A=A,
                AT=AT,
                train_pair_mask_tensor=train_pair_mask_tensor,
                mir_feat=mir_feat,
                drug_feat=drug_feat,
                HMM=HMM,
                HDD=HDD,
                obj=obj,
            )

            if (
                epoch == 0
                or (epoch + 1) % args.log_interval == 0
                or epoch == args.epochs - 1
            ):
                print(
                    f"[Fold {fold_id}] Epoch {epoch + 1:04d}/{args.epochs} | "
                    f"L={losses['loss']:.5f} | "
                    f"Lmain={losses['loss_main']:.5f} | "
                    f"Lhgcl={losses['loss_hgcl']:.5f} | "
                    f"Ldisen={losses['loss_disen']:.5f}"
                )

        score_matrix = infer_score(
            model=model,
            A=A,
            AT=AT,
            mir_feat=mir_feat,
            drug_feat=drug_feat,
            HMM=HMM,
            HDD=HDD,
            obj=obj,
        )

        metrics, curve_data = evaluate_samples(
            score_matrix=score_matrix,
            test_samples=test_samples,
        )

        row = {
            "Fold": fold_id,
            "Train_Pos": tr_pos,
            "Train_Neg": tr_neg,
            "Test_Pos": te_pos,
            "Test_Neg": te_neg,
            **metrics,
        }
        fold_rows.append(row)

        print(
            f"[Fold {fold_id}] "
            f"AUC={metrics['AUC']:.5f} | "
            f"AUPR={metrics['AUPR']:.5f} | "
            f"ACC={metrics['Accuracy']:.5f} | "
            f"PRE={metrics['Precision']:.5f} | "
            f"REC={metrics['Recall']:.5f} | "
            f"SPE={metrics['Specificity']:.5f} | "
            f"F1={metrics['F1']:.5f} | "
            f"MCC={metrics['MCC']:.5f}"
        )

        pred_df = pd.DataFrame({
            "miRNA_index": test_samples[:, 0].astype(np.int64),
            "drug_index": test_samples[:, 1].astype(np.int64),
            "label": curve_data["y_true"],
            "score": curve_data["y_score"],
        })
        pred_df.to_csv(
            os.path.join(args.result_dir, f"fold_{fold_id}_predictions.csv"),
            index=False,
        )

        torch.save(
            model.state_dict(),
            os.path.join(args.result_dir, f"model_fold_{fold_id}.pth"),
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(
        os.path.join(args.result_dir, "fold_metrics.csv"),
        index=False,
    )

    metric_cols = [
        "AUC",
        "AUPR",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "MCC",
    ]

    summary_rows = []
    print("\n" + "=" * 80)
    print("Final 5-fold results (VCMDS-style balanced pair CV)")
    print("=" * 80)

    for col in metric_cols:
        values = fold_df[col].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        summary_rows.append({
            "Metric": col,
            "Mean": mean,
            "Std": std,
        })
        print(f"{col:12s}: {mean:.5f} ± {std:.5f}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.result_dir, "summary_mean_std.csv"),
        index=False,
    )

    print("=" * 80)
    print(f"[Saved] {args.result_dir}/balanced_samples_seed{args.seed}.npy")
    print(f"[Saved] {args.result_dir}/vcmds_kfold_seed{args.seed}.npz")
    print(f"[Saved] {args.result_dir}/fold_metrics.csv")
    print(f"[Saved] {args.result_dir}/summary_mean_std.csv")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "miRNA-drug association prediction using the current model "
            "under a VCMDS-style balanced 5-fold protocol."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset1",
        help=(
            "Directory containing guanlianmatrix.txt, m-mmatrix.txt, "
            "and d-d2matrix.txt"
        ),
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="mirna_drug_vcmds_results",
    )

    parser.add_argument("--epochs", type=int, default=135)
    parser.add_argument("--cv_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--sim_weight", type=float, default=0.3)

    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--mir_similarity_knn", type=int, default=50)
    parser.add_argument("--drug_similarity_knn", type=int, default=30)

    parser.add_argument("--linear_rank", type=int, default=80)
    parser.add_argument("--glhmf_alpha", type=float, default=0.8)
    parser.add_argument("--glhmf_k", type=int, default=10)
    parser.add_argument("--glhmf_lambda", type=float, default=0.0001)
    parser.add_argument("--glhmf_max_iter", type=int, default=1000)

    parser.add_argument("--lambda_main", type=float, default=1.0)
    parser.add_argument("--lambda_hgcl", type=float, default=0.5)
    parser.add_argument("--lambda_disen", type=float, default=0.3)

    parser.add_argument("--log_interval", type=int, default=10)

    args = parser.parse_args()

    if args.cv_num != 5:
        print(
            f"[Warning] VCMDS reference protocol uses 5-fold CV, "
            f"but cv_num={args.cv_num}."
        )

    if not (0.0 <= args.sim_weight <= 1.0):
        raise ValueError("--sim_weight must be in [0, 1].")

    main(args)
