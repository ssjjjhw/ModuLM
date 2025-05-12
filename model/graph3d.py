import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
from torch_geometric.nn import GATConv,GINConv,GCNConv
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class EGNNLayer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, coord, edge_index):
        row, col = edge_index
        dx = coord[row] - coord[col]
        dist = torch.norm(dx, dim=-1, keepdim=True)

        edge_feat = torch.cat([x[row], x[col], dist], dim=-1)
        msg = self.edge_mlp(edge_feat)
        coord_update = dx * self.coord_mlp(msg)

        delta_coord = torch.zeros_like(coord)
        delta_coord.index_add_(0, row, coord_update)
        coord = coord + delta_coord

        node_msg = torch.zeros_like(x)
        node_msg.index_add_(0, row, msg)
        x = x + self.node_mlp(node_msg)

        return x, coord


class SimpleEGNNModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList([
            EGNNLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        src_tokens,
        padded_coordinates,
        src_distance,      # 保留参数但不使用
        src_edge_type      # 保留参数但不使用
    ):
        B, N = src_tokens.shape
        device = src_tokens.device

        # === Padding mask ===
        padding_mask = src_tokens.eq(self.padding_idx)  # [B, N]

        # === Embedding ===
        x = self.embed_tokens(src_tokens)               # [B, N, D]
        coord = padded_coordinates                      # [B, N, 3]

        # === Flatten for batching ===
        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)  # [B*N]

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False, max_num_neighbors=32)

        for layer in self.layers:
            x, coord = layer(x, coord, edge_index)

        # === Scatter back ===
        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask

class RBFExpansion(nn.Module):
    def __init__(self, num_kernels=128, cutoff=6.0, gamma=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.gamma = gamma
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_kernels), requires_grad=False)

    def forward(self, distances):
        # distances: [E, 1]
        diff = distances - self.centers.to(distances.device)  # [E, K]
        return torch.exp(-self.gamma * diff ** 2)  # [E, K]


class SchNetInteractionBlock(nn.Module):
    def __init__(self, hidden_dim=512, rbf_dim=128):
        super().__init__()
        self.mlp_edge = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mlp_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_rbf):
        row, col = edge_index
        edge_feature = self.mlp_edge(edge_rbf)  # [E, D]
        msg = edge_feature * x[col]  # [E, D]
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)
        x = x + self.mlp_node(agg)
        return x


class SimpleSchNetModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512, rbf_kernels=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.rbf = RBFExpansion(num_kernels=rbf_kernels, cutoff=6.0)
        self.layers = nn.ModuleList([
            SchNetInteractionBlock(embed_dim, rbf_kernels)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        src_tokens,
        padded_coordinates,
        src_distance,       # optional, not used directly
        src_edge_type       # optional, not used
    ):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)  # [B, N]
        x = self.embed_tokens(src_tokens)               # [B, N, D]
        coord = padded_coordinates                      # [B, N, 3]

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False, max_num_neighbors=32)
        row, col = edge_index
        dist = torch.norm(coord[row] - coord[col], dim=-1, keepdim=True)  # [E, 1]
        rbf_feature = self.rbf(dist)  # [E, K]

        for layer in self.layers:
            x = layer(x, edge_index, rbf_feature)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask
    

# class RBFExpansion(nn.Module):
#     def __init__(self, num_kernels=64, cutoff=6.0, gamma=10.0):
#         super().__init__()
#         self.cutoff = cutoff
#         self.gamma = gamma
#         self.centers = nn.Parameter(torch.linspace(0, cutoff, num_kernels), requires_grad=False)

#     def forward(self, dist):
#         diff = dist - self.centers.to(dist.device)
#         return torch.exp(-self.gamma * diff ** 2)


def compute_angle(v1, v2, eps=1e-8):
    v1_norm = F.normalize(v1, dim=-1)
    v2_norm = F.normalize(v2, dim=-1)
    cos_angle = (v1_norm * v2_norm).sum(-1).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cos_angle)
    return angle.unsqueeze(-1)  # [E, 1]


class DimeNetBlock(nn.Module):
    def __init__(self, embed_dim, rbf_dim, angle_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim + angle_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, coord, rbf_feature):
        row, col = edge_index
        v_ij = coord[row] - coord[col]

        # 找三元组 (k, j, i)
        # 其中 (k, j) ∈ edge_index, (j, i) ∈ edge_index
        # 我们寻找中间点 j 的所有相邻边组成角度
        # 暂简化为仅依赖 v_ij 本身角度方向（非完整 DimeNet++）

        angle = compute_angle(v_ij, -v_ij)  # dummy (shape: [E, 1])
        edge_feat = torch.cat([rbf_feature, angle], dim=-1)
        edge_msg = self.edge_mlp(edge_feat)

        out = torch.zeros_like(x)
        out.index_add_(0, row, edge_msg)
        x = x + self.update_mlp(out)
        return x


class SimpleDimeNetPlusPlusModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512, rbf_dim=64, angle_dim=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.rbf = RBFExpansion(num_kernels=rbf_dim)
        self.layers = nn.ModuleList([
            DimeNetBlock(embed_dim, rbf_dim, angle_dim)
            for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)  # [B, N]
        x = self.embed_tokens(src_tokens)               # [B, N, D]
        coord = padded_coordinates                      # [B, N, 3]

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False, max_num_neighbors=32)
        row, col = edge_index
        dist = torch.norm(coord[row] - coord[col], dim=-1, keepdim=True)
        rbf_feature = self.rbf(dist)  # [E, K]

        for layer in self.layers:
            x = layer(x, edge_index, coord, rbf_feature)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask
    
class PaiNNInteraction(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear_q = nn.Linear(dim, dim)
        self.linear_kv = nn.Linear(dim, dim * 3)
        self.vec_mlp = nn.Linear(dim, dim)

    def forward(self, s, v, edge_index):
        row, col = edge_index

        q = self.linear_q(s)
        kv = self.linear_kv(s)
        k, v_s, v_v = kv.chunk(3, dim=-1)

        edge_weight = (q[row] * k[col]).sum(-1, keepdim=True)
        edge_msg_s = edge_weight * v_s[col]
        edge_msg_v = edge_weight * v_v[col].unsqueeze(-1) * v[col].unsqueeze(1)

        agg_s = torch.zeros_like(s)
        agg_s.index_add_(0, row, edge_msg_s)

        agg_v = torch.zeros_like(v)
        agg_v.index_add_(0, row, edge_msg_v.squeeze(-1))

        s = s + agg_s
        v = v + self.vec_mlp(agg_v)
        return s, v


class SimplePaiNNModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, dim=512):
        super().__init__()
        self.dim = dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), dim, padding_idx=self.padding_idx)

        self.vec_proj = nn.Linear(3, dim)
        self.interactions = nn.ModuleList([
            PaiNNInteraction(dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)
        x = self.embed_tokens(src_tokens)  # [B, N, D]
        coord = padded_coordinates         # [B, N, 3]

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        v = self.vec_proj(coord)  # [N, D]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.interactions:
            x, v = layer(x, v, edge_index)

        out = torch.zeros(B * N, self.dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.dim)

        return encoder_rep, padding_mask
    
class SE3TransformerLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.coord_mlp = nn.Linear(1, dim)

    def forward(self, x, coord, edge_index):
        row, col = edge_index
        dx = coord[row] - coord[col]
        dist = torch.norm(dx, dim=-1, keepdim=True)

        q = self.query(x[row])
        k = self.key(x[col])
        v = self.value(x[col])

        attn = (q * k).sum(-1, keepdim=True)
        weight = F.softmax(attn, dim=0)

        delta_x = weight * v
        delta_coord = weight * self.coord_mlp(dist) * dx

        x_out = torch.zeros_like(x)
        x_out.index_add_(0, row, delta_x)

        coord_out = torch.zeros_like(coord)
        coord_out.index_add_(0, row, delta_coord)

        x = x + x_out
        coord = coord + coord_out
        return x, coord


class SimpleSE3TransformerModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, dim=512):
        super().__init__()
        self.dim = dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            SE3TransformerLayer(dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)
        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            x, coord = layer(x, coord, edge_index)

        out = torch.zeros(B * N, self.dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.dim)

        return encoder_rep, padding_mask
    
class GVP(nn.Module):
    def __init__(self, dim_s, dim_v):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(dim_s, dim_s),
            nn.SiLU(),
            nn.Linear(dim_s, dim_s)
        )
        self.vector_proj = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.SiLU(),
            nn.Linear(dim_v, dim_v)
        )

    def forward(self, s, v):
        s_out = self.scalar_proj(s)
        v_out = self.vector_proj(v)
        return s_out, v_out


class GVPMessagePassing(nn.Module):
    def __init__(self, dim_s, dim_v):
        super().__init__()
        self.edge_gvp = GVP(dim_s * 2, dim_v)
        self.update_gvp = GVP(dim_s, dim_v)

    def forward(self, s, v, coord, edge_index):
        row, col = edge_index
        ds = torch.cat([s[row], s[col]], dim=-1)
        dv = coord[row] - coord[col]

        # Basic edge vector features
        dist = torch.norm(dv, dim=-1, keepdim=True)
        edge_scalar, edge_vector = self.edge_gvp(ds, dv.unsqueeze(-1))

        # Message passing
        agg_s = torch.zeros_like(s)
        agg_v = torch.zeros_like(v)

        agg_s.index_add_(0, row, edge_scalar)
        agg_v.index_add_(0, row, edge_vector)

        s_out, v_out = self.update_gvp(agg_s, agg_v)
        return s + s_out, v + v_out


class SimpleGVPGNNModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, dim_s=512, dim_v=16):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), dim_s, padding_idx=self.padding_idx)

        self.init_vec = nn.Linear(3, dim_v)
        self.layers = nn.ModuleList([
            GVPMessagePassing(dim_s, dim_v) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)
        s = self.embed_tokens(src_tokens)               # [B, N, D_s]
        coord = padded_coordinates                      # [B, N, 3]

        s = s.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        s = s[mask]
        coord = coord[mask]
        batch = batch[mask]
        v = self.init_vec(coord)  # initial vector features

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            s, v = layer(s, v, coord, edge_index)

        out = torch.zeros(B * N, s.size(-1), device=device)
        out[mask] = s
        encoder_rep = out.view(B, N, s.size(-1))
        return encoder_rep, padding_mask
    
class SimpleGeoFormerModel(nn.Module):
    def __init__(self, dictionary, embed_dim=512, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            activation='gelu',
            dropout=0.1,
            batch_first=True
        )
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.dist_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)  # scalar bias
        )

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device

        padding_mask = src_tokens.eq(self.padding_idx)
        x = self.embed_tokens(src_tokens)  # [B, N, D]
        dist = src_distance.unsqueeze(-1)  # [B, N, N, 1]
        attn_bias = self.dist_proj(dist).squeeze(-1)  # [B, N, N]

        # fill padding bias to large negative
        attn_bias = attn_bias.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        attn_bias = attn_bias.masked_fill(padding_mask.unsqueeze(2), float("-inf"))

        encoder_rep = self.encoder(x, src_key_padding_mask=padding_mask, mask=attn_bias)
        return encoder_rep, padding_mask
    
class GINLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.conv = GINConv(self.mlp)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class SimpleGIN_ESMModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList([GINLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            x = layer(x, edge_index)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask

class GearNetLayer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.edge_mlp = nn.Linear(1, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, coord, edge_index):
        row, col = edge_index
        dx = coord[row] - coord[col]
        dist = torch.norm(dx, dim=-1, keepdim=True)
        edge_attr = self.edge_mlp(dist)

        edge_msg = torch.cat([x[col], edge_attr], dim=-1)
        msg = self.mlp(edge_msg)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)

        return x + agg


class SimpleGearNetModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            GearNetLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            x = layer(x, coord, edge_index)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)
        return encoder_rep, padding_mask

class GCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = GCNConv(dim, dim)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class SimpleGCN_ESMModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            GCNLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)               # [B, N, D]
        coord = padded_coordinates                      # [B, N, 3]

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            x = layer(x, edge_index)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask
    
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.conv = GATConv(dim, dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class SimpleGAT_ESMModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            GATLayer(embed_dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)

        for layer in self.layers:
            x = layer(x, edge_index)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)

        return encoder_rep, padding_mask

from torch_scatter import scatter_add


class RBF(nn.Module):
    def __init__(self, num_kernels=64, cutoff=6.0, gamma=10.0):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_kernels), requires_grad=False)
        self.gamma = gamma

    def forward(self, dist):
        diff = dist - self.centers.to(dist.device)  # [E, K]
        return torch.exp(-self.gamma * diff ** 2)


class AngleEncoding(nn.Module):
    def __init__(self, angle_dim=16):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, angle_dim),
            nn.SiLU(),
            nn.Linear(angle_dim, angle_dim)
        )

    def forward(self, cos_angle):
        return self.linear(cos_angle.unsqueeze(-1))  # [E, angle_dim]


class SphereNetLayer(nn.Module):
    def __init__(self, embed_dim=512, rbf_dim=64, angle_dim=16):
        super().__init__()
        self.edge_proj = nn.Linear(rbf_dim + angle_dim, embed_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, coord, edge_index, rbf_feature, angle_feature):
        row, col = edge_index
        edge_feat = torch.cat([rbf_feature, angle_feature], dim=-1)
        msg = self.edge_proj(edge_feat)  # [E, D]

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)
        x = x + self.node_proj(agg)
        return x


class SimpleSphereNetModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512, rbf_dim=64, angle_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.rbf = RBF(num_kernels=rbf_dim)
        self.angle_proj = AngleEncoding(angle_dim=angle_dim)

        self.layers = nn.ModuleList([
            SphereNetLayer(embed_dim, rbf_dim, angle_dim)
            for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)
        row, col = edge_index
        edge_vec = coord[row] - coord[col]
        dist = torch.norm(edge_vec, dim=-1)  # [E]

        # === RBF encoding
        rbf_feature = self.rbf(dist)  # [E, K]

        # === Angle encoding: use triplets (k, j, i)
        # 近似计算 cos θ: j-i vs k-j
        angle_feature = torch.zeros_like(rbf_feature[:, :16])
        try:
            adj_list = [[] for _ in range(x.shape[0])]
            for i, j in zip(row.tolist(), col.tolist()):
                adj_list[i].append(j)

            angle_list = []
            for i in range(x.shape[0]):
                neighbors = adj_list[i]
                for m in range(len(neighbors)):
                    for n in range(m + 1, len(neighbors)):
                        j, k = neighbors[m], neighbors[n]
                        v1 = coord[i] - coord[j]
                        v2 = coord[i] - coord[k]
                        cos_theta = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).clamp(-1, 1)
                        angle_list.append(cos_theta)

            angle_tensor = torch.stack(angle_list) if angle_list else torch.zeros(1, device=device)
            angle_feature = self.angle_proj(angle_tensor)
        except:
            pass  # fallback to zero

        for layer in self.layers:
            x = layer(x, coord, edge_index, rbf_feature, angle_feature)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)
        return encoder_rep, padding_mask

class GS_RBF(nn.Module):
    def __init__(self, num_kernels=64, cutoff=6.0, gamma=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.gamma = gamma
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_kernels), requires_grad=False)

    def forward(self, dist):
        diff = dist - self.centers.to(dist.device)
        return torch.exp(-self.gamma * diff ** 2)


class GSAngleEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, angle_cos):
        return self.mlp(angle_cos.unsqueeze(-1))  # [E, D]


class GSphereNetLayer(nn.Module):
    def __init__(self, embed_dim=512, rbf_dim=64, angle_dim=32):
        super().__init__()
        self.edge_proj = nn.Linear(rbf_dim + angle_dim, embed_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, rbf_feature, angle_feature):
        row, col = edge_index
        edge_feat = torch.cat([rbf_feature, angle_feature], dim=-1)
        msg = self.edge_proj(edge_feat)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)
        return x + self.update_mlp(agg)


class SimpleGSphereNetModel(nn.Module):
    def __init__(self, dictionary, num_layers=4, embed_dim=512, rbf_dim=64, angle_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), embed_dim, padding_idx=self.padding_idx)

        self.rbf = GS_RBF(rbf_dim)
        self.angle_encoder = GSAngleEncoder(angle_dim)
        self.layers = nn.ModuleList([
            GSphereNetLayer(embed_dim, rbf_dim, angle_dim) for _ in range(num_layers)
        ])

    def forward(self, src_tokens, padded_coordinates, src_distance, src_edge_type):
        B, N = src_tokens.shape
        device = src_tokens.device
        padding_mask = src_tokens.eq(self.padding_idx)

        x = self.embed_tokens(src_tokens)
        coord = padded_coordinates

        x = x.view(B * N, -1)
        coord = coord.view(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        mask = ~padding_mask.view(-1)

        x = x[mask]
        coord = coord[mask]
        batch = batch[mask]

        edge_index = radius_graph(coord, r=6.0, batch=batch, loop=False)
        row, col = edge_index
        edge_vec = coord[row] - coord[col]
        dist = torch.norm(edge_vec, dim=-1)
        rbf_feature = self.rbf(dist)

        # === Build angle feature from (k, j, i)
        angle_feature = torch.zeros_like(rbf_feature[:, :32])
        try:
            adj = [[] for _ in range(x.shape[0])]
            for r, c in zip(row.tolist(), col.tolist()):
                adj[r].append(c)

            angle_cos_list = []
            for j in range(x.shape[0]):
                neighbors = adj[j]
                for i_idx in range(len(neighbors)):
                    for k_idx in range(i_idx + 1, len(neighbors)):
                        i = neighbors[i_idx]
                        k = neighbors[k_idx]
                        vec_ij = coord[i] - coord[j]
                        vec_kj = coord[k] - coord[j]
                        cos_angle = F.cosine_similarity(vec_ij.unsqueeze(0), vec_kj.unsqueeze(0)).clamp(-1, 1)
                        angle_cos_list.append(cos_angle)

            angle_tensor = torch.stack(angle_cos_list) if angle_cos_list else torch.zeros(1, device=device)
            angle_feature = self.angle_encoder(angle_tensor)
        except:
            pass

        for layer in self.layers:
            x = layer(x, edge_index, rbf_feature, angle_feature)

        out = torch.zeros(B * N, self.embed_dim, device=device)
        out[mask] = x
        encoder_rep = out.view(B, N, self.embed_dim)
        return encoder_rep, padding_mask