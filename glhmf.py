import numpy as np
from scipy.linalg import inv


class GLHMF:
    def __init__(self, rank=50, lamb=0.0001, max_iter=1000, alpha=0.8, k=10, mu=0.0001):
        self.r = rank
        self.lamb = lamb
        self.mu = mu
        self.max_iter = max_iter
        self.alpha = alpha
        self.k = k

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
        U_svd, singular_values, Vt_svd = np.linalg.svd(H_het, full_matrices=False)
        r = min(self.r, len(singular_values))
        sqrt_singular_values = np.diag(np.sqrt(singular_values[:r]))
        U_s = U_svd[:, :r] @ sqrt_singular_values
        V_s = Vt_svd[:r, :].T @ sqrt_singular_values
        regularizer = (self.lamb + self.mu) * np.eye(r)

        for _ in range(self.max_iter):
            U_s = (H_het @ V_s + self.mu * V_s) @ inv(V_s.T @ V_s + regularizer)
            V_s = (H_het.T @ U_s + self.mu * U_s) @ inv(U_s.T @ U_s + regularizer)

        return U_s @ V_s.T

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