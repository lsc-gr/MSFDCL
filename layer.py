import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class LightGCNConv(nn.Module):
    def __init__(self, embed_dim, bias=True, use_residual=True):
        super(LightGCNConv, self).__init__()
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        if bias:
            self.bias = Parameter(torch.DoubleTensor(embed_dim))
            stdv = 1. / math.sqrt(embed_dim)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        self.sparse_mask = None

    def forward(self, x, adj):
        x = x.double()
        if self.sparse_mask is not None:
            adj = self.sparse_mask(adj)
        out = adj.matmul(x)
        if self.bias is not None:
            out = out + self.bias
        if self.use_residual:
            out = out + x
        return out


class LightGCN(nn.Module):

    def __init__(self, n_node, embed_dim, n_layers=2, dropout=0.5, use_residual=True,
                 attention_type='simple'):
        super(LightGCN, self).__init__()
        self.n_node = n_node
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.attention_type = attention_type
        self.dropout = nn.Dropout(dropout)

        # 初始嵌入
        self.node_emb = Parameter(torch.DoubleTensor(n_node, embed_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # 特征投影层
        self.proj = nn.Linear(n_node, embed_dim, dtype=torch.double)
        nn.init.xavier_uniform_(self.proj.weight)

        # 卷积层
        self.use_residual = use_residual
        self.convs = nn.ModuleList([LightGCNConv(embed_dim=embed_dim) for _ in range(n_layers)])

        # ✅ 注意力机制（二选一，推荐simple）
        if attention_type == 'simple':
            # 简单注意力：直接学习每层权重（参数少，稳定，易解释）
            self.layer_attn = nn.Parameter(torch.ones(n_layers + 1))
            self.attention_mlp = None
        elif attention_type == 'mlp':
            # MLP注意力：基于嵌入内容动态计算权重（复杂，可能过拟合）
            self.layer_attn = None
            self.attention_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1, bias=False)
            ).double()
        else:
            raise ValueError("attention_type must be 'simple' or 'mlp'")

    def forward(self, adj, feat=None):
        """
        前向传播：支持两种注意力机制
        """
        # 获取初始嵌入
        if feat is not None:
            x = self.proj(feat.double())
        else:
            x = self.node_emb
        x = self.dropout(x)

        all_embeddings = [x]
        for conv in self.convs:
            x = conv(x, adj)
            all_embeddings.append(x)

        # ✅ 步骤3：注意力加权融合（修复重复计算问题）
        if self.attention_type == 'simple':
            # 简单版本：直接学习每层权重（推荐）
            attn_weights = F.softmax(self.layer_attn, dim=0)  # [n_layers+1]
            attn_weights = attn_weights.view(1, -1, 1)  # [1, n_layers+1, 1]
            embeddings_stack = torch.stack(all_embeddings, dim=1)
            final_emb = torch.sum(embeddings_stack * attn_weights, dim=1)

        else:  # 'mlp'
            # MLP版本：基于嵌入内容动态计算（复杂，慎用）
            attn_scores = []
            for emb in all_embeddings:
                pooled = emb.mean(dim=0, keepdim=True)  # 全局池化 [1, embed_dim]
                score = self.attention_mlp(pooled)  # [1, 1]
                attn_scores.append(score)
            attn_weights = F.softmax(torch.cat(attn_scores, dim=0).squeeze(), dim=0)
            embeddings_stack = torch.stack(all_embeddings, dim=1)
            final_emb = torch.sum(embeddings_stack * attn_weights.view(1, -1, 1), dim=1)

        return final_emb


class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = Parameter(torch.zeros(size=(num_in_edge, num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = Parameter(torch.zeros(num_hidden, dtype=torch.double))

    def forward(self, H_T):
        return self.act(torch.mm(H_T, self.W1) + self.b1)


class node_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = Parameter(torch.zeros(size=(num_in_edge, num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = Parameter(torch.zeros(num_hidden, dtype=torch.double))

    def forward(self, H):
        return self.act(H.mm(self.W1) + 2 * self.b1)


class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        return self.act(self.dropout(z_node).mm(self.dropout(z_hyperedge).t()))


class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z_node, z_hyperedge):
        return self.act(z_node.mm(z_hyperedge.t()))