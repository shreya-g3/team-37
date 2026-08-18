import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


class SpatialBranch(nn.Module):
    """GATv2 stack over the physical spot-coordinate graph."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, heads=4, edge_dim=1, dropout=0.2):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, edge_dim=edge_dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        return F.elu(h)


class ExpressionBranch(nn.Module):
    """GraphSAGE stack over the SVD-similarity expression graph."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.relu(h)


class FusionHead(nn.Module):
    """Light MLP fusing the two branch embeddings into protein z-score predictions."""

    def __init__(self, spatial_dim=64, expr_dim=64, hidden_dim=64, n_proteins=44, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spatial_dim + expr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_proteins),
        )

    def forward(self, h_spatial, h_expr):
        h = torch.cat([h_spatial, h_expr], dim=-1)
        return self.net(h)


class DualBranchGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        spatial_hidden=128,
        expr_hidden=128,
        branch_out=64,
        fusion_hidden=64,
        n_proteins=44,
        heads=4,
        edge_dim=1,
        dropout=0.2,
    ):
        super().__init__()
        self.spatial_branch = SpatialBranch(
            in_dim, hidden_dim=spatial_hidden, out_dim=branch_out,
            heads=heads, edge_dim=edge_dim, dropout=dropout,
        )
        self.expr_branch = ExpressionBranch(
            in_dim, hidden_dim=expr_hidden, out_dim=branch_out, dropout=dropout,
        )
        self.head = FusionHead(
            spatial_dim=branch_out, expr_dim=branch_out,
            hidden_dim=fusion_hidden, n_proteins=n_proteins, dropout=dropout,
        )

    def forward(self, x, spatial_edge_index, spatial_edge_attr, expr_edge_index):
        h_spatial = self.spatial_branch(x, spatial_edge_index, spatial_edge_attr)
        h_expr = self.expr_branch(x, expr_edge_index)
        return self.head(h_spatial, h_expr)


if __name__ == "__main__":
    # wiring sketch using the outputs of graph_construction.py
    from graph_construction import fit_truncated_svd, build_spatial_graph, build_expression_graph
    from preprocess import preprocess_rna  # your existing function

    rna_train = preprocess_rna("rna_train.h5ad")
    svd, latent_train = fit_truncated_svd(rna_train, n_components=100)

    spatial_edge_index, spatial_edge_attr = build_spatial_graph(rna_train, k=6)
    expr_edge_index, _ = build_expression_graph(latent_train, k=10)  # SAGEConv ignores edge_weight

    x = torch.tensor(latent_train, dtype=torch.float32)  # or your combined feature matrix

    model = DualBranchGNN(in_dim=x.shape[1], n_proteins=44)
    preds = model(x, spatial_edge_index, spatial_edge_attr, expr_edge_index)  # (n_spots, 44)