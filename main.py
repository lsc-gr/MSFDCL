import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from torch import optim

from glhmf import GLHMF
from hgcl import channel_hgcl_loss, hgcl_loss
from hypergraph_utils import (
    construct_H_with_KNN, generate_G_from_H, knn_local_enhance,
)
from model import Model


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def integrated_similarity(profiles, external, weight):
    n = profiles.shape[0]
    if external.shape != (n, n):
        raise ValueError(f"Expected similarity shape {(n, n)}, got {external.shape}")

    norm_sum = np.sum(np.square(np.linalg.norm(profiles, axis=1)))
    gamma = n / max(norm_sum, 1e-12)
    diff = profiles[:, None, :] - profiles[None, :, :]
    gip = np.exp(-gamma * np.sum(diff * diff, axis=2))

    sim = weight * external + (1.0 - weight) * gip
    return (sim + sim.T) / 2.0


def balanced_samples(matrix, seed):
    positives = np.argwhere(matrix != 0)
    unknown = np.argwhere(matrix == 0)

    if len(unknown) < len(positives):
        raise ValueError("Not enough unobserved pairs for 1:1 sampling.")

    rng = np.random.RandomState(seed)
    ids = rng.choice(len(unknown), size=len(positives), replace=False)

    positive_rows = np.column_stack((positives, np.ones(len(positives), dtype=int)))
    negative_rows = np.column_stack((unknown[ids], np.zeros(len(ids), dtype=int)))
    return np.vstack((positive_rows, negative_rows)).astype(np.int64)


def similarity_graph(sim, k, device):
    k = min(k, max(1, sim.shape[0] - 1))
    local = knn_local_enhance(sim, k=k, thresh=0.0)
    incidence = construct_H_with_KNN(local)
    return generate_G_from_H(incidence).double().to(device)


def association_graph(profiles, device):
    incidence = construct_H_with_KNN(profiles)
    propagation = generate_G_from_H(incidence).double()
    edges = torch.from_numpy(np.array(np.where(propagation.cpu().numpy() > 0.1)))
    return incidence, propagation.to(device), edges.long().to(device)


def rank_features(values, rank, device):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] < rank:
        values = np.pad(values, ((0, 0), (0, rank - values.shape[1])))
    return torch.from_numpy(values[:, :rank]).to(device)


def build_inputs(matrix, mir_sim, drug_sim, args, device):
    HMM = similarity_graph(mir_sim, args.mir_similarity_knn, device)
    HDD = similarity_graph(drug_sim, args.drug_similarity_knn, device)

    factorizer = GLHMF(
        rank=args.linear_rank,
        lamb=args.glhmf_lambda,
        max_iter=args.glhmf_max_iter,
        alpha=args.glhmf_alpha,
        k=args.glhmf_k,
    )
    global_matrix, local_matrix, mir_g, drug_g, mir_l, drug_l = (
        factorizer.fit_transform(matrix, mir_sim, drug_sim)
    )

    mir_inc, HMD_G, mir_edges_g = association_graph(global_matrix, device)
    _, HMD_L, mir_edges_l = association_graph(local_matrix, device)
    drug_inc, HDM_G, drug_edges_g = association_graph(global_matrix.T, device)
    _, HDM_L, drug_edges_l = association_graph(local_matrix.T, device)

    A = torch.from_numpy(matrix).double().to(device)
    mir_feat = torch.eye(matrix.shape[0], dtype=A.dtype, device=device)
    drug_feat = torch.eye(matrix.shape[1], dtype=A.dtype, device=device)

    inputs = (
        A.t().contiguous(), A,
        mir_feat, drug_feat, HMM, HDD,
        HMD_G, HMD_L, HDM_G, HDM_L,
        mir_edges_g, mir_edges_l, drug_edges_g, drug_edges_l,
        rank_features(mir_g, args.linear_rank, device),
        rank_features(mir_l, args.linear_rank, device),
        rank_features(drug_g, args.linear_rank, device),
        rank_features(drug_l, args.linear_rank, device),
    )
    return A, inputs, mir_inc, drug_inc


def train(model, optimizer, inputs, mask, labels, pos_weight, mir_inc, drug_inc, args):
    model.train()

    (
        sim_score, global_score, local_score, _, final_score,
        _, mir_global, mir_local, _,
        _, drug_local, drug_global, _,
        _, _, _, _, _,
        mir_factor_g, mir_factor_l, drug_factor_g, drug_factor_l,
    ) = model(*inputs)

    loss_main = sum(
        F.binary_cross_entropy_with_logits(
            score.t()[mask], labels, pos_weight=pos_weight
        )
        for score in (sim_score, global_score, local_score, final_score)
    )

    loss_hgcl = labels.new_zeros(())
    if mir_local is not None and drug_global is not None:
        loss_hgcl = (
            hgcl_loss(mir_global, mir_local, mir_inc, tau=0.1)
            + hgcl_loss(drug_global, drug_local, drug_inc, tau=0.1)
        )

    loss_disen = labels.new_zeros(())
    for global_factors, local_factors, incidence in (
        (mir_factor_g, mir_factor_l, mir_inc),
        (drug_factor_g, drug_factor_l, drug_inc),
    ):
        if global_factors is not None and local_factors is not None:
            loss_disen = loss_disen + channel_hgcl_loss(
                global_factors, local_factors, incidence,
                k=2, tau=0.1, ortho_weight=0.05,
            )

    loss = (
        args.lambda_main * loss_main
        + args.lambda_hgcl * loss_hgcl
        + args.lambda_disen * loss_disen
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return tuple(x.item() for x in (loss, loss_main, loss_hgcl, loss_disen))


@torch.no_grad()
def evaluate(model, inputs, test_samples):
    model.eval()
    scores = model(*inputs)[4].t().cpu().numpy()

    rows, cols, labels = test_samples.T
    scores = scores[rows, cols]

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    threshold = 0.5
    if thresholds.size:
        p, r = precision[:-1], recall[:-1]
        f1_curve = 2 * p * r / (p + r + 1e-12)
        threshold = thresholds[np.nanargmax(f1_curve)]

    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, _, _ = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

    return {
        "AUC": roc_auc_score(labels, scores),
        "AUPR": auc(recall, precision),
        "Accuracy": accuracy_score(labels, predicted),
        "Precision": precision_score(labels, predicted, zero_division=0),
        "Recall": recall_score(labels, predicted, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "F1": f1_score(labels, predicted, zero_division=0),
        "MCC": matthews_corrcoef(labels, predicted),
        "Threshold": float(threshold),
    }


def main(args):
    set_seed(args.seed)
    if not 0 <= args.sim_weight <= 1:
        raise ValueError("--sim_weight must be in [0, 1].")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    root = Path(args.data_dir)

    matrix = np.loadtxt(root / "guanlianmatrix.txt", dtype=np.float64)
    mir_external = np.loadtxt(root / "m-mmatrix.txt", dtype=np.float64)
    drug_external = np.loadtxt(root / "d-d2matrix.txt", dtype=np.float64)


    mir_sim = integrated_similarity(matrix, mir_external, args.sim_weight)
    drug_sim = integrated_similarity(matrix.T, drug_external, args.sim_weight)
    samples = balanced_samples(matrix, args.seed)
    cv = KFold(n_splits=args.cv_num, shuffle=True, random_state=args.seed)
    A, inputs, mir_inc, drug_inc = build_inputs(matrix, mir_sim, drug_sim, args, device)

    print(f"Device: {device} | miRNAs: {A.shape[0]} | drugs: {A.shape[1]}")
    fold_results = []

    for fold, (train_ids, test_ids) in enumerate(cv.split(samples), start=1):
        set_seed(args.seed + fold)
        train_samples, test_samples = samples[train_ids], samples[test_ids]
        rows, cols, train_y = train_samples.T

        mask_np = np.zeros(matrix.shape, dtype=bool)
        mask_np[rows, cols] = True
        mask = torch.from_numpy(mask_np).to(device)
        labels = A[mask]

        n_pos = int(np.sum(train_y == 1))
        n_neg = int(np.sum(train_y == 0))
        pos_weight = A.new_tensor([n_neg / max(n_pos, 1)])

        print(f"Fold {fold}/{args.cv_num} | train: {len(train_ids)} | test: {len(test_ids)}")

        model = Model(
            num_in_node=A.shape[1],
            num_in_edge=A.shape[0],
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            linear_rank=args.linear_rank,
            gamma=args.gamma,
        ).double().to(device)

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(1, args.epochs + 1):
            losses = train(
                model, optimizer, inputs, mask, labels,
                pos_weight, mir_inc, drug_inc, args,
            )
            if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
                total, main_loss, hgcl, disen = losses
                print(
                    f"Epoch {epoch:04d} | loss {total:.5f} | main {main_loss:.5f} "
                    f"| hgcl {hgcl:.5f} | disen {disen:.5f}"
                )

        metrics = evaluate(model, inputs, test_samples)
        fold_results.append(metrics)
        print(" | ".join(f"{name}: {value:.5f}" for name, value in metrics.items()))

    print(f"\n{args.cv_num}-fold results (mean ± sample SD)")
    for name in fold_results[0]:
        if name != "Threshold":
            values = [fold[name] for fold in fold_results]
            print(f"{name:12s}: {np.mean(values):.5f} ± {np.std(values, ddof=1):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset1")
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

    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--lambda_main", type=float, default=1.0)
    parser.add_argument("--lambda_hgcl", type=float, default=0.5)
    parser.add_argument("--lambda_disen", type=float, default=0.3)
    parser.add_argument("--log_interval", type=int, default=10)

    main(parser.parse_args())