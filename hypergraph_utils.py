import numpy as np
import torch
import torch.nn as nn

def Eu_dis(x):

    # 安全转换（避免视图共享）
    x = np.asmatrix(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def cos_dis(x):

    x = np.asmatrix(x)
    norm = np.linalg.norm(x, axis=1)
    norm[norm == 0] = 1
    x = np.divide(x, norm.reshape(-1, 1))
    dis_mat = 1 - np.matmul(x, x.T)
    return dis_mat


def feature_concat(normal_col=False, *F_list):

    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        # 跳过 None 或空列表
        if h is None or (isinstance(h, (list, tuple)) and len(h) == 0):
            continue

        # 如果 h 是列表，但里面只有一个元素，就把它取出来
        # 这是一个鲁棒性处理
        if isinstance(h, (list, tuple)) and len(h) == 1:
            h = h[0]

        # 如果 H 还是 None，直接把当前的 h 赋值给 H
        if H is None:
            H = h
        else:
            # 核心修改：确保 H 和 h 都是数组再进行拼接
            # 如果 H 是列表，把它转换成数组
            if isinstance(H, list):
                H = np.array(H)
            # 如果 h 是列表，把它转换成数组
            if isinstance(h, list):
                h = np.array(h)

            # 现在可以安全地使用 hstack 了
            H = np.hstack((H, h))
    return H

def generate_G_from_H(H, variable_weight=False):

    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):

    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    a = []
    for i in DE:
        if i == 0:
            a.append(0)
        else:
            a.append(np.power(i, -1))
    invDE = np.asmatrix(np.diag(a))

    b = []
    for i in DV:
        if i == 0:
            b.append(0)
        else:
            b.append(np.power(i, -0.5))
    DV2 = np.asmatrix(np.diag(b))

    W = np.asmatrix(np.diag(W))
    H = np.asmatrix(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
    G = torch.Tensor(G)
    # D_V1 = torch.Tensor(D_V1)
    return G # , D_V1


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):

    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        # nearest_idx = np.array(np.argsort(dis_vec)).squeeze() #选距离小的
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze() #选大的
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, K_neigs=[5], split_diff_scale=False, is_probH=True, m_prob=1):

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    # dis_mat = Eu_dis(X)
    dis_mat = cos_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def knn_local_enhance(sim_mat: np.ndarray, k: int, thresh: float = 0.0) -> np.ndarray:

    n = sim_mat.shape[0]
    knn_adj = np.zeros_like(sim_mat)

    # 1. 行方向 top-k
    for i in range(n):
        top_k_idx = np.argsort(sim_mat[i, :])[-(k+1):-1]  # 去掉自己
        knn_adj[i, top_k_idx] = sim_mat[i, top_k_idx]

    # 2. 对称化（无向图）
    knn_adj = np.maximum(knn_adj, knn_adj.T)

    # 3. 阈值（可选）
    if thresh > 0:
        knn_adj = (knn_adj >= thresh).astype(float)

    return knn_adj


# 在 hypergraph_utils.py 末尾追加
def construct_meta_path_adj(relationship_dict, meta_path_edges, threshold=0.0):

    import scipy.sparse as sp

    # 矩阵连乘
    adj = None
    for edge_type in meta_path_edges:
        if adj is None:
            # 首段关系
            adj = relationship_dict[edge_type]
        else:
            # 矩阵乘法传播高阶关系
            next_adj = relationship_dict[edge_type]
            # 处理稀疏矩阵×稠密矩阵的情况
            if sp.issparse(adj):
                adj = adj @ next_adj
            else:
                adj = adj @ next_adj

    # 阈值处理（稀疏化）
    if sp.issparse(adj):
        adj = adj.todense()
    adj = np.array(adj)
    adj[adj < threshold] = 0

    # 对称化（保证无向图）
    adj = np.maximum(adj, adj.T)

    # 转换为torch tensor
    return torch.from_numpy(adj).float()

