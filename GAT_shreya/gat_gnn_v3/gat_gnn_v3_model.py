import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


# ---------------------------------------------------------
# Global constants
# ---------------------------------------------------------
GAT_PROJ_DIM     = 128   # project node features to this before GATv2
GAT_HEADS        = 4
GAT_OUT_PER_HEAD = 32    # gat_hidden = GAT_HEADS * GAT_OUT_PER_HEAD


class SpatialBranch(nn.Module):
    """
    GATv2 stack over the physical spot-coordinate graph.
    """

    def __init__(self, in_dim,
                 proj_dim=GAT_PROJ_DIM,
                 heads=GAT_HEADS,
                 out_per_head=GAT_OUT_PER_HEAD,
                 dropout=0.2,
                 edge_dropout=0.1):
        super().__init__()

        # pre-projection: compress node features before any graph op
        self.pre_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ELU(),
        )

        gat_hidden = heads * out_per_head

        self.gat1 = GATv2Conv(
            in_channels=proj_dim, out_channels=out_per_head,
            heads=heads, concat=True, dropout=dropout, edge_dim=1,
        )
        self.norm1 = nn.LayerNorm(gat_hidden)

        self.gat2 = GATv2Conv(
            in_channels=gat_hidden, out_channels=out_per_head,
            heads=heads, concat=True, dropout=dropout, edge_dim=1,
        )

        self.out_dim = gat_hidden
        self.dropout = dropout
        self.edge_dropout = edge_dropout

    def _maybe_drop_edges(self, edge_index, edge_attr):
        if not self.training or self.edge_dropout <= 0:
            return edge_index, edge_attr
        E = edge_index.size(1)
        keep = torch.rand(E, device=edge_index.device) > self.edge_dropout
        return edge_index[:, keep], edge_attr[keep]

    def forward(self, x, edge_index, edge_attr):
        ea = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr

        h = self.pre_proj(x)

        ei1, ea1 = self._maybe_drop_edges(edge_index, ea)
        h = self.gat1(h, ei1, edge_attr=ea1)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # re-sample which edges dropped out
        ei2, ea2 = self._maybe_drop_edges(edge_index, ea)
        h = self.gat2(h, ei2, edge_attr=ea2)
        return F.elu(h)


class ResidualSAGEBlock(nn.Module):
    """SAGEConv -> LayerNorm -> ELU(h + x) residual -> Dropout."""

    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.conv = SAGEConv(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.elu(h + x)
        return F.dropout(h, p=self.dropout, training=self.training)


class ExpressionBranch(nn.Module):
    """
    Residual GraphSAGE stack over the SVD-similarity expression graph.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.sage_blocks = nn.ModuleList(
            [ResidualSAGEBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_weight):
        x_w = self._weighted_agg(x, edge_index, edge_weight)
        x_combined = torch.cat([x, x_w], dim=1)
        h = self.input_proj(x_combined)
        for blk in self.sage_blocks:
            h = blk(h, edge_index)
        return h

    @staticmethod
    def _weighted_agg(x, edge_index, edge_weight):
        """Weighted mean of neighbour features, using expression-graph edge_weight."""
        src, dst = edge_index
        weighted = x[dst] * edge_weight.unsqueeze(1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, src.unsqueeze(1).expand_as(weighted), weighted)
        wsum = torch.zeros(x.shape[0], 1, device=x.device)
        wsum.scatter_add_(0, src.unsqueeze(1), edge_weight.unsqueeze(1))
        return out / wsum.clamp(min=1e-8)


class FusionBlock(nn.Module):
    """
    Bilinear-gated fusion of the two branches, replacing plain
    concat([h_spatial, h_expr]) -> MLP.

    Three signals feed the gate instead of one:
      1. h_spatial, h_expr projected to a common `dim` (linear signal)
      2. their elementwise product (a bilinear interaction term -- lets
         the gate react to agreement/disagreement patterns between the
         two views, not just their raw magnitudes; plain concat can't
         represent "these two branches actively disagree on this node")
      3. a residual pass-through path (raw concat projected separately)
         so the head still has direct access to both branches even where
         the gate saturates toward 0 or 1 - a pure gate can zero out a
         branch's contribution entirely for a node where the gate is
         wrong; the residual keeps some signal flowing regardless.

    fused = gate * hs + (1 - gate) * he + residual_proj(concat)
    then LayerNorm + ELU + Dropout, so the head receives a normalized,
    bounded input regardless of how the gate/residual paths scale.
    """

    def __init__(self, spatial_dim, expr_dim, dim, dropout=0.2):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, dim)
        self.expr_proj = nn.Linear(expr_dim, dim)

        self.gate = nn.Sequential(
            nn.Linear(spatial_dim + expr_dim + dim, dim),  # concat + interaction term
            nn.ELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.residual_proj = nn.Linear(spatial_dim + expr_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout
        self.out_dim = dim

    def forward(self, h_spatial, h_expr):
        hs = self.spatial_proj(h_spatial)
        he = self.expr_proj(h_expr)
        interaction = hs * he

        gate_in = torch.cat([h_spatial, h_expr, interaction], dim=1)
        g = self.gate(gate_in)

        gated = g * hs + (1 - g) * he
        residual = self.residual_proj(torch.cat([h_spatial, h_expr], dim=1))

        fused = self.norm(gated + residual)
        fused = F.elu(fused)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        return fused, g


class DualBranchGNN(nn.Module):
    """
    Branch A - GATv2 on spatial k-NN graph (distance-weighted attention, edge dropout)
    Branch B - Residual SAGEConv on expression k-NN graph (similarity-weighted pre-agg)
    Fusion   - bilinear-gated FusionBlock -> MLP -> protein predictions

    in_dim is set from the node feature matrix passed in (SVD latent).
    """

    def __init__(self, in_dim, out_dim,
                 sage_hidden=128,
                 n_sage_layers=2,
                 dropout=0.2,
                 gat_proj_dim=GAT_PROJ_DIM,
                 gat_heads=GAT_HEADS,
                 gat_out_per_head=GAT_OUT_PER_HEAD,
                 edge_dropout=0.1,
                 fusion_dim=None):
        super().__init__()

        self.spatial_branch = SpatialBranch(
            in_dim, proj_dim=gat_proj_dim, heads=gat_heads,
            out_per_head=gat_out_per_head, dropout=dropout, edge_dropout=edge_dropout,
        )
        self.expr_branch = ExpressionBranch(
            in_dim, hidden_dim=sage_hidden, num_layers=n_sage_layers, dropout=dropout,
        )

        fusion_dim = fusion_dim or max(self.spatial_branch.out_dim, self.expr_branch.out_dim)
        self.fusion = FusionBlock(
            self.spatial_branch.out_dim, self.expr_branch.out_dim, fusion_dim, dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, sage_hidden // 2),
            nn.LayerNorm(sage_hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(sage_hidden // 2, out_dim),
        )

        self._config = dict(
            in_dim=in_dim, out_dim=out_dim,
            sage_hidden=sage_hidden, n_sage_layers=n_sage_layers,
            dropout=dropout, gat_proj_dim=gat_proj_dim,
            gat_heads=gat_heads, gat_out_per_head=gat_out_per_head,
            edge_dropout=edge_dropout, fusion_dim=fusion_dim,
        )

    def forward(self, x, spatial_ei, spatial_ew, expr_ei, expr_ew, return_gate=False):
        h_spatial = self.spatial_branch(x, spatial_ei, spatial_ew)
        h_expr = self.expr_branch(x, expr_ei, expr_ew)
        fused, gate = self.fusion(h_spatial, h_expr)
        preds = self.head(fused)
        if return_gate:
            return preds, gate
        return preds


# ---------------------------------------------------------
# Loss functions
# ---------------------------------------------------------

def pearson_loss(yp, yt, valid=None, eps=1e-8):
    """
    1 - per-protein Pearson correlation, averaged across proteins.
    If `valid` is given (bool, same shape as yt, True = usable entry),
    mean/covariance/variance are computed only over valid rows per protein,
    rather than zero-filling missing entries first (which would bias the
    correlation toward zero-vs-zero similarity for any protein with
    non-trivial missingness).
    """
    if valid is None:
        vp = yp - yp.mean(0, keepdim=True)
        vt = yt - yt.mean(0, keepdim=True)
        r = (vp * vt).sum(0) / ((vp ** 2).sum(0) * (vt ** 2).sum(0) + eps).sqrt()
        return (1 - r).mean()

    valid_f = valid.float()
    n_valid = valid_f.sum(0).clamp(min=1.0)

    mean_p = (yp * valid_f).sum(0) / n_valid
    mean_t = (yt * valid_f).sum(0) / n_valid

    cp = (yp - mean_p) * valid_f
    ct = (yt - mean_t) * valid_f

    cov = (cp * ct).sum(0)
    var_p = (cp ** 2).sum(0)
    var_t = (ct ** 2).sum(0)
    r = cov / (var_p * var_t + eps).sqrt()

    enough = valid.sum(0) >= 2
    if enough.sum() == 0:
        return torch.tensor(0.0, device=yp.device)
    return (1 - r)[enough].mean()


def combined_loss(yp, yt, w=0.8, missing_mask=None):
    """
    w * SmoothL1 + (1 - w) * pearson_loss.
    missing_mask (bool, same shape as yt, True = originally-missing protein
    value) excludes those entries from BOTH terms consistently.
    """
    if missing_mask is not None:
        valid = ~missing_mask
        reg = F.smooth_l1_loss(yp, yt, reduction="none")
        reg = (reg * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        pear = pearson_loss(yp, yt, valid=valid)
    else:
        reg = F.smooth_l1_loss(yp, yt)
        pear = pearson_loss(yp, yt)
    return w * reg + (1 - w) * pear