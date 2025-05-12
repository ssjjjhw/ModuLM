from math import ceil

# from layer import *
from model.hetognn.layer import *


class DenseGINConv2d(nn.Module):
    """GIN Layer for 2D dense input with Linear-based MLP."""

    def __init__(self, input_dim, output_dim, groups, eps=0.0, train_eps=True):
        super(DenseGINConv2d, self).__init__()
        self.groups = groups

        # Linear-based MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        # Learnable epsilon for self-loops
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps, dtype=torch.float32))
        else:
            self.register_buffer('eps', torch.tensor(eps, dtype=torch.float32))

    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, F = x.size()
        G = adj.size(0)
        assert G == 1, "GIN expects single graph adjacency per layer."

        # Adjust dimensions for single graph
        adj = adj.squeeze(0)  # [N, N]

        # Normalize adjacency matrix and add self-loops
        adj_norm = adj + torch.eye(N, device=adj.device)  # Add self-loops
        neighbor_info = torch.matmul(adj_norm, x)  # [B, C, N, F]

        # Combine with self-node information
        out = (1 + self.eps) * x + neighbor_info  # [B, C, N, F]

        # Reshape for Linear layers in MLP
        out = out.view(B * C * N, F)  # Flatten into [B * C * N, F]
        # print(out.shape)
        # Pass through MLP
        out = self.mlp(out)  # [B * C * N, output_dim]

        # Reshape back to original dimensions
        out = out.view(B, C, N, -1)  # [B, C, N, F]

        return out



class GNNStack(nn.Module):
    """GIN-based GNN Stack."""

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):
        super().__init__()
        
        # Initialization remains unchanged
        self.num_nodes = 2 * num_nodes
        self.num_graphs = groups
        self.g_constr = multi_shallow_embedding(num_nodes, self.num_nodes, self.num_graphs)
        self.g_constr_heto = multi_shallow_embedding_heto(num_nodes, self.num_nodes, self.num_graphs)
        
        gnn_model, heads = self.build_gnn_model(gnn_model_type)
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [(k - 1) // 2 for k in kern_size]
        
        # self.tconvs = nn.ModuleList(
        #     [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] + 
        #     [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer+1]), padding=(0, paddings[layer+1])) for layer in range(num_layers - 2)] + 
        #     [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        # )
        
        self.gconvs = nn.ModuleList(
            [gnn_model(512, 512, groups)] + 
            [gnn_model(512, 512, groups) for _ in range(num_layers - 2)] + 
            [gnn_model(512, 512, groups)]
        )
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(512)] + 
            [nn.BatchNorm2d(512) for _ in range(num_layers - 2)] + 
            [nn.BatchNorm2d(512)]
        )
        
        # self.diffpool = nn.ModuleList(
        #     [Dense_TimeDiffPool2d(num_nodes, num_nodes, kern_size[layer], paddings[layer]) for layer in range(num_layers)]
        # )
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)
        # self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_agg = nn.Linear(256, 1)

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1

    def forward(self, inputs: Tensor, heto=False):
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs

        adj = self.g_constr(x.device) if not heto else self.g_constr_heto(x.device)

        for tconv, gconv, bn in zip(self.tconvs, self.gconvs, self.bns):
            x = gconv(x,adj)
            # print(x.shape)
            # x, adj = (gconv(tconv(x), adj), adj)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        aggregated = x.squeeze(1)
        return aggregated



