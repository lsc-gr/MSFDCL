import torch
import torch.nn as nn

class SparseMask(nn.Module):
    """
    可学习稀疏掩码，Top-k硬阈值，端到端训练
    k：保留边比例，默认5%
    """
    def __init__(self, n_nodes, init_sparsity=0.05):
        super().__init__()
        # 可学习参数，初始全1（不抑制任何边）
        self.M = nn.Parameter(torch.ones(n_nodes, n_nodes))
        self.sparsity = init_sparsity

    def forward(self, adj):
        """
        adj: [N, N] 原始邻接矩阵（0/1或float）
        return: [N, N] 掩码后的邻接矩阵（同shape）
        """
        M = torch.sigmoid(self.M)          # 0-1
        k = int(self.sparsity * M.numel())
        if k > 0:
            threshold = M.view(-1).topk(k)[0][-1]
            mask = (M >= threshold).float()
        else:
            mask = torch.zeros_like(M)
        return adj * mask.detach()         # 硬阈值不可导