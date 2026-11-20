import torch
import torch.nn.functional as F


def hgcl_loss(z_anchor, z_pos, H, tau=0.1):

    if z_anchor is None or z_pos is None:
        return torch.tensor(0.0, device=z_anchor.device if z_anchor is not None else 'cpu')

    z_a = F.normalize(z_anchor, dim=1)
    z_p = F.normalize(z_pos, dim=1)

    sim_matrix = torch.mm(z_a, z_p.t()) / tau
    exp_sim = torch.exp(sim_matrix)

    if not torch.is_tensor(H):
        H = torch.from_numpy(H).float().to(z_a.device)
    else:
        H = H.float().to(z_a.device)

    pos_mask = (H @ H.t() > 0).float()
    pos_mask.fill_diagonal_(1.0)  # 自身也是正样本

    # 分子：正样本相似度之和（包含自身）
    pos_sim = (exp_sim * pos_mask).sum(dim=1)

    # 分母：所有样本相似度之和（包含自身，与论文公式一致）
    all_sim = exp_sim.sum(dim=1)

    loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8)).mean()
    return loss



def channel_hgcl_loss(z_anchor_dis, z_pos_dis, H, k=2, tau=0.1, ortho_weight=0.05):

    if z_anchor_dis is None or z_pos_dis is None:
        return torch.tensor(0.0, device=z_anchor_dis.device if z_anchor_dis is not None else 'cpu')

    N, total_dim = z_anchor_dis.shape
    x_dim = total_dim // k

    # reshape 为 [N, k, x_dim]
    z_a = z_anchor_dis.view(N, k, x_dim)
    z_p = z_pos_dis.view(N, k, x_dim)

    # 超边掩码
    if not torch.is_tensor(H):
        H = torch.from_numpy(H).float().to(z_a.device)
    else:
        H = H.float().to(z_a.device)

    pos_mask = (H @ H.t() > 0).float()
    pos_mask.fill_diagonal_(1.0)

    # 逐通道对比损失
    channel_loss = 0.0
    for c in range(k):
        z_ac = F.normalize(z_a[:, c, :], dim=1)
        z_pc = F.normalize(z_p[:, c, :], dim=1)

        sim = torch.mm(z_ac, z_pc.t()) / tau
        exp_sim = torch.exp(sim)

        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        channel_loss += -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8)).mean()

    channel_loss = channel_loss / k


    ortho_loss = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            sim_ij = torch.sum(z_a[:, i, :] * z_a[:, j, :], dim=1)
            ortho_loss += torch.mean(sim_ij ** 2)

    if k > 1:
        ortho_loss = ortho_loss / (k * (k - 1) / 2.0)
    else:
        ortho_loss = torch.tensor(0.0, device=z_a.device)

    return channel_loss + ortho_weight * ortho_loss