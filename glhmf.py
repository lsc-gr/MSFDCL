import numpy as np
from scipy.linalg import inv


class GLHMF:


    def __init__(self, rank=50, lamb=0.0001, max_iter=1000, alpha=0.8, k=10):
        self.r = rank          # 截断秩（对应论文参数 r）
        self.lamb = lamb       # 正则化系数（对应论文 λ）
        self.max_iter = max_iter
        self.alpha = alpha     # 全局标签传播阻尼系数
        self.k = k             # 局部 KNN 邻居数

    def _global_similarity(self, S):

        n = S.shape[0]
        I = np.eye(n)
        col_sum = S.sum(axis=0)
        col_sum[col_sum == 0] = 1e-10
        S_bar = S / col_sum
        inv_mat = inv(I - self.alpha * S_bar)
        return (1 - self.alpha) * inv_mat

    def _local_similarity(self, S):

        n = S.shape[0]
        result = np.zeros_like(S)
        for i in range(n):
            neighbors = np.argsort(S[i])[::-1][:self.k]
            denom = S[i, neighbors].sum()
            if denom > 0:
                result[i, neighbors] = S[i, neighbors] / denom
        return result

    def _als_factorization(self, H_het):

        U, S, Vt = np.linalg.svd(H_het, full_matrices=False)
        S = np.sqrt(S)
        r = min(self.r, len(S))
        Wt = np.diag(S[:r])
        U = U[:, :r]
        V = Vt[:r, :].T
        A = U @ Wt
        B = V @ Wt

        for _ in range(self.max_iter):
            A = H_het @ B @ inv(B.T @ B + self.lamb * np.eye(r))
            B = H_het.T @ A @ inv(A.T @ A + self.lamb * np.eye(r))

        return A @ B.T

    def fit_transform(self, Y, SM, SD):

        n_m, n_d = Y.shape

        GSM = self._global_similarity(SM)
        GSD = self._global_similarity(SD)

        LSM = self._local_similarity(SM)
        LSD = self._local_similarity(SD)


        H_global = np.block([[GSM, Y], [Y.T, GSD]])
        L_global = self._als_factorization(H_global)
        Y_global = L_global[:n_m, n_m:]


        H_local = np.block([[LSM, Y], [Y.T, LSD]])
        L_local = self._als_factorization(H_local)
        Y_local = L_local[:n_m, n_m:]


        U_g, S_g, Vt_g = np.linalg.svd(Y_global, full_matrices=False)
        r_g = min(self.r, len(S_g))
        mir_g = U_g[:, :r_g] * np.sqrt(S_g[:r_g])          # [n_m, r_g]
        dis_g = Vt_g[:r_g, :].T * np.sqrt(S_g[:r_g])      # [n_d, r_g]

        U_l, S_l, Vt_l = np.linalg.svd(Y_local, full_matrices=False)
        r_l = min(self.r, len(S_l))
        mir_l = U_l[:, :r_l] * np.sqrt(S_l[:r_l])          # [n_m, r_l]
        dis_l = Vt_l[:r_l, :].T * np.sqrt(S_l[:r_l])      # [n_d, r_l]

        return Y_global, Y_local, mir_g, dis_g, mir_l, dis_l
