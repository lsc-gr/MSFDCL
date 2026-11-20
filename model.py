from layer import *
from sparse_mask import SparseMask
import torch
import torch.nn as nn
from torch_geometric.data import Data
import random
from copy import deepcopy
import torch.nn.functional as F
class HypergraphConv(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=True).double()
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        nn.init.zeros_(self.linear.bias)
        self.sparse_mask = None

    def forward(self, x, adj):
        x = x.double()
        if self.sparse_mask is not None:
            adj = self.sparse_mask(adj)
        out = adj.matmul(x)
        out = self.linear(out)
        return F.relu(out)


class HGCN(nn.Module):

    def __init__(self, n_node, embed_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_node = n_node
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(n_node, embed_dim, dtype=torch.double)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.convs = nn.ModuleList([HypergraphConv(embed_dim) for _ in range(n_layers)])

    def forward(self, adj, feat=None):
        if feat is not None:
            x = self.proj(feat.double())
        else:
            eye = torch.eye(self.n_node, dtype=torch.double, device=adj.device)
            x = self.proj(eye)
        x = self.dropout(x)

        for conv in self.convs:
            x = conv(x, adj)
            x = self.dropout(x)
        return x

class HeterogeneousFusionGate(nn.Module):
    def __init__(self, embed_dim, num_types=3):
        super(HeterogeneousFusionGate, self).__init__()
        self.num_types = num_types
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_types, num_types, bias=True).double(),
            nn.Softmax(dim=-1)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.gate[0].weight, gain=1.414)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, feat_linear, feat_gcn, feat_disen):
        concat = torch.cat([feat_linear, feat_gcn, feat_disen], dim=-1)
        weights = self.gate(concat)
        fused = (weights[:, 0:1] * feat_linear +
                 weights[:, 1:2] * feat_gcn +
                 weights[:, 2:3] * feat_disen)
        return fused, weights


class DisenEncoder(nn.Module):
    def __init__(self, in_dim, k=2, x_dim=256):
        super(DisenEncoder, self).__init__()
        self.k = k
        self.routit = 3
        self.x_dim = x_dim
        self.linear = nn.Linear(in_dim, self.k * self.x_dim).double()
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)

    def forward(self, x, src_trg):
        x = self.linear(x)
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k

        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)
        c = x

        for t in range(self.routit):
            p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)
            p = F.softmax(p, dim=1)
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)
            weight_sum = (p * z).view(m, d)
            c = c.index_add_(0, trg, weight_sum)
            c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)

        return c


class SemanticAttention(nn.Module):
    def __init__(self, in_dim, attn_vec_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, attn_vec_dim, bias=True).double()
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False).double()
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, metapath_views):
        beta = []
        for view in metapath_views:
            graph_repr = torch.mean(view, dim=0)
            hidden = torch.tanh(self.fc1(graph_repr))
            attn_score = self.fc2(hidden)
            beta.append(attn_score)

        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)

        stacked = torch.stack(metapath_views, dim=0)
        fused = torch.sum(beta.view(len(metapath_views), 1, 1) * stacked, dim=0)
        return fused, beta


class Model(nn.Module):
    def __init__(self,
                 num_in_node=100,
                 num_in_edge=561,
                 embed_dim=512,
                 n_layers=2,
                 dropout=0.5,
                 sparsity=0.05,
                 linear_rank=50):
        super().__init__()
        self.decoder1 = decoder1(act=lambda x: x)
        self.circ_mask = SparseMask(num_in_edge, sparsity)
        self.dis_mask = SparseMask(num_in_node, sparsity)

        self.gcn_edge = LightGCN(n_node=num_in_edge, embed_dim=embed_dim,
                                 n_layers=n_layers, dropout=dropout)
        self.gcn_node = LightGCN(n_node=num_in_node, embed_dim=embed_dim,
                                 n_layers=n_layers, dropout=dropout)

        for layer in self.gcn_edge.convs:
            layer.sparse_mask = self.circ_mask
        for layer in self.gcn_node.convs:
            layer.sparse_mask = self.dis_mask

        self.semantic_attn_circ = SemanticAttention(embed_dim, attn_vec_dim=128)
        self.semantic_attn_dis = SemanticAttention(embed_dim, attn_vec_dim=128)

        self.disen_circ = DisenEncoder(embed_dim, k=2, x_dim=embed_dim // 2)
        self.disen_dis = DisenEncoder(embed_dim, k=2, x_dim=embed_dim // 2)

        self.proj_linear_circ = nn.Linear(linear_rank, embed_dim, bias=True).double()
        self.proj_linear_dis = nn.Linear(linear_rank, embed_dim, bias=True).double()
        nn.init.xavier_uniform_(self.proj_linear_circ.weight, gain=1.414)
        nn.init.xavier_uniform_(self.proj_linear_dis.weight, gain=1.414)
        nn.init.zeros_(self.proj_linear_circ.bias)
        nn.init.zeros_(self.proj_linear_dis.bias)

        self.fusion_gate_circ = HeterogeneousFusionGate(embed_dim, num_types=3)
        self.fusion_gate_dis = HeterogeneousFusionGate(embed_dim, num_types=3)

    def forward(self, AT, A,
                circ_feat, dis_feat,
                HMM, HDD,
                HMD_G, HMD_L, HDM_G, HDM_L,
                edge_index_md_g=None, edge_index_md_l=None,
                edge_index_dm_g=None, edge_index_dm_l=None,
                circ_linear_g=None, circ_linear_l=None,
                dis_linear_g=None, dis_linear_l=None):

        circ_view_mm = self.gcn_edge(adj=HMM, feat=circ_feat)
        circ_view_md_g = self.gcn_edge(adj=HMD_G, feat=circ_feat)
        circ_view_md_l = self.gcn_edge(adj=HMD_L, feat=circ_feat) if HMD_L is not None else None


        circ_g_dis = self.disen_circ(circ_view_md_g, edge_index_md_g) \
            if edge_index_md_g is not None else None
        circ_l_dis = self.disen_circ(circ_view_md_l, edge_index_md_l) \
            if edge_index_md_l is not None else None


        views_circ = [circ_view_mm, circ_view_md_g]
        if circ_view_md_l is not None:
            views_circ.append(circ_view_md_l)
        circ_fused, w_circ = self.semantic_attn_circ(views_circ)


        dis_view_dd = self.gcn_node(adj=HDD, feat=dis_feat)
        dis_view_dm_l = self.gcn_node(adj=HDM_L, feat=dis_feat)
        dis_view_dm_g = self.gcn_node(adj=HDM_G, feat=dis_feat) if HDM_G is not None else None


        dis_l_dis = self.disen_dis(dis_view_dm_l, edge_index_dm_l) \
            if edge_index_dm_l is not None else None
        dis_g_dis = self.disen_dis(dis_view_dm_g, edge_index_dm_g) \
            if edge_index_dm_g is not None else None

        views_dis = [dis_view_dd, dis_view_dm_l]
        if dis_view_dm_g is not None:
            views_dis.append(dis_view_dm_g)
        dis_fused, w_dis = self.semantic_attn_dis(views_dis)


        if circ_linear_g is not None and circ_linear_l is not None:
            circ_linear = self.proj_linear_circ((circ_linear_g + circ_linear_l) / 2.0)
        elif circ_linear_g is not None:
            circ_linear = self.proj_linear_circ(circ_linear_g)
        elif circ_linear_l is not None:
            circ_linear = self.proj_linear_circ(circ_linear_l)
        else:
            circ_linear = circ_fused

        if dis_linear_g is not None and dis_linear_l is not None:
            dis_linear = self.proj_linear_dis((dis_linear_g + dis_linear_l) / 2.0)
        elif dis_linear_g is not None:
            dis_linear = self.proj_linear_dis(dis_linear_g)
        elif dis_linear_l is not None:
            dis_linear = self.proj_linear_dis(dis_linear_l)
        else:
            dis_linear = dis_fused


        if circ_g_dis is not None and circ_l_dis is not None:
            circ_disen = (circ_g_dis + circ_l_dis) / 2.0
        elif circ_g_dis is not None:
            circ_disen = circ_g_dis
        elif circ_l_dis is not None:
            circ_disen = circ_l_dis
        else:
            circ_disen = circ_fused

        if dis_g_dis is not None and dis_l_dis is not None:
            dis_disen = (dis_g_dis + dis_l_dis) / 2.0
        elif dis_g_dis is not None:
            dis_disen = dis_g_dis
        elif dis_l_dis is not None:
            dis_disen = dis_l_dis
        else:
            dis_disen = dis_fused


        circ_final, gate_w_circ = self.fusion_gate_circ(circ_linear, circ_fused, circ_disen)
        dis_final, gate_w_dis = self.fusion_gate_dis(dis_linear, dis_fused, dis_disen)


        reconstructionMM = self.decoder1(dis_view_dd, circ_view_mm)
        reconstructionMD_G = self.decoder1(dis_view_dm_g, circ_view_md_g) if dis_view_dm_g is not None else None
        reconstructionMD_L = self.decoder1(dis_view_dm_l, circ_view_md_l) if circ_view_md_l is not None else None


        result = self.decoder1(dis_final, circ_final)
        recover = result


        return (reconstructionMM, reconstructionMD_G, reconstructionMD_L, result, recover,
                circ_view_mm, circ_view_md_g, circ_view_md_l, None,
                dis_view_dd, dis_view_dm_l, dis_view_dm_g, None,
                circ_final, dis_final, None,
                w_circ, w_dis,
                circ_g_dis, circ_l_dis, dis_g_dis, dis_l_dis)
