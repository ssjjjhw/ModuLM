import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, x1, x2):
        B, N, D = x1.size()
        score = self.bilinear(x1.view(-1, D), x2.view(-1, D))  # [B*N, 1]
        score = score.view(B, N, 1)
        attn_weights = torch.sigmoid(score)
        return attn_weights * x1 + (1 - attn_weights) * x2

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x_query, x_context):
        out, _ = self.attn(x_query, x_context, x_context)
        return out

class Highway(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        T = torch.sigmoid(self.gate(x))
        H = F.relu(self.transform(x))
        return T * H + (1 - T) * x

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        fused = torch.cat([x1, x2], dim=-1)  # [B, N, 2D]
        gate = self.gate(fused)
        fused_proj = self.proj(fused)
        return gate * fused_proj + (1 - gate) * x1

class BilinearFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim)

    def forward(self, x1, x2):
        B, N, D = x1.size()
        out = self.bilinear(x1.view(-1, D), x2.view(-1, D)).view(B, N, D)
        return out

