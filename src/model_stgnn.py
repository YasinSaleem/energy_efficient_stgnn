#!/usr/bin/env python3
"""
Energy-Efficient Spatio-Temporal GNN (ST-GNN) Model

- Spatial GCN using precomputed normalized adjacency
- Temporal GRU over time dimension
- Multi-step forecasting: predicts next H hours for all nodes

Input X: [B, T_in, N, 1]
Output Y: [B, H, N]

Author: Energy-Efficient STGNN Project
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ADJ_PATH = PROCESSED_DIR / "adjacency_matrix.npz"
NODE_MAP_PATH = PROCESSED_DIR / "node_map.json"


# ============================================================================
# UTILS: LOAD GRAPH
# ============================================================================

def load_adjacency_and_num_nodes():
    """
    Load normalized adjacency matrix (csr npz) and infer num_nodes.
    Returns:
        adj_torch: torch.sparse_coo_tensor [N, N]
        num_nodes: int
    """
    if not ADJ_PATH.exists():
        raise FileNotFoundError(f"Adjacency file not found: {ADJ_PATH}")

    if not NODE_MAP_PATH.exists():
        raise FileNotFoundError(f"Node map not found: {NODE_MAP_PATH}")

    with NODE_MAP_PATH.open("r") as f:
        node_map = json.load(f)
    num_nodes = len(node_map)

    print(f"[graph] Loading adjacency from {ADJ_PATH}")
    adj_csr = load_npz(ADJ_PATH)

    if adj_csr.shape[0] != num_nodes or adj_csr.shape[1] != num_nodes:
        raise ValueError(
            f"Adjacency shape {adj_csr.shape} does not match num_nodes={num_nodes}"
        )

    coo = adj_csr.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data.astype(np.float32)

    indices_t = torch.from_numpy(indices).long()
    values_t = torch.from_numpy(values)

    adj_torch = torch.sparse_coo_tensor(
        indices_t,
        values_t,
        size=(num_nodes, num_nodes),
        dtype=torch.float32
    ).coalesce()

    print(f"[graph] Adjacency: nnz={adj_torch._nnz()} | shape={adj_torch.shape}")
    return adj_torch, num_nodes


# ============================================================================
# SPATIAL GCN LAYER
# ============================================================================

class SpatialGCNLayer(nn.Module):
    """
    Simple GCN layer using pre-normalized adjacency.

    Input:  x [B, T, N, Fin]
    Output: y [B, T, N, Fout]
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: Optional[nn.Module] = nn.ReLU(),
                 dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x:   [B, T, N, Fin]
        adj: sparse [N, N]
        """
        B, T, N, Fin = x.shape
        # Linear on features
        h = self.lin(x)  # [B, T, N, Fout]

        # Sparse adjacency multiplication:
        # reshape to [N, B*T*Fout], then A @ X_flat
        N2 = h.shape[2]
        assert N2 == N, "Node dimension mismatch"

        h_perm = h.permute(2, 0, 1, 3)       # [N, B, T, F]
        h_flat = h_perm.reshape(N, -1)       # [N, B*T*F]

        Ah_flat = torch.sparse.mm(adj, h_flat)  # [N, B*T*F]

        Ah_perm = Ah_flat.reshape(N, B, T, -1)  # [N, B, T, F]
        out = Ah_perm.permute(1, 2, 0, 3)       # [B, T, N, F]

        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        return out


# ============================================================================
# ST-GNN MODEL
# ============================================================================

class STGNNModel(nn.Module):
    """
    Spatio-Temporal GNN Model:

    - Spatial: 1 or 2 GCN layers on each time step's graph snapshot
    - Temporal: GRU over time for each node
    - Readout: MLP to map GRU hidden state -> HORIZON future steps

    Input X: [B, T, N, 1]
    Output:  [B, H, N]
    """

    def __init__(self,
                 num_nodes: int,
                 in_features: int = 1,
                 gcn_hidden: int = 16,
                 gcn_layers: int = 1,
                 gru_hidden: int = 32,
                 horizon: int = 6,
                 dropout: float = 0.1,
                 adj: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_features = in_features
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.horizon = horizon

        # Adjacency as buffer so it moves with .to(device)
        if adj is None:
            adj, _ = load_adjacency_and_num_nodes()
        self.register_buffer("adjacency", adj)  # sparse [N, N]

        # Spatial GCN stack
        spatial_layers = []
        fin = in_features
        for i in range(gcn_layers):
            fout = gcn_hidden
            spatial_layers.append(
                SpatialGCNLayer(fin, fout, activation=nn.ReLU(), dropout=dropout)
            )
            fin = fout
        self.spatial_gcn = nn.ModuleList(spatial_layers)

        # Temporal GRU over time for each node
        # We'll reshape [B, T, N, F] -> [B*N, T, F]
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Readout: hidden state -> HORIZON
        self.fc_out = nn.Linear(gru_hidden, horizon)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, N, 1]
        returns: [B, H, N]
        """
        B, T, N, Fin = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"

        h = x  # [B, T, N, 1]

        # ----- Spatial GCN -----
        for layer in self.spatial_gcn:
            h = layer(h, self.adjacency)  # [B, T, N, gcn_hidden]

        # ----- Temporal GRU -----
        # Reshape to [B*N, T, F]
        B, T, N, Fg = h.shape
        h_seq = h.permute(0, 2, 1, 3).reshape(B * N, T, Fg)  # [B*N, T, Fg]

        gru_out, h_final = self.gru(h_seq)  # h_final: [1, B*N, gru_hidden]
        h_last = h_final[-1]                # [B*N, gru_hidden]

        h_last = self.dropout(h_last)

        # ----- Readout -----
        out = self.fc_out(h_last)  # [B*N, horizon]
        out = out.view(B, N, self.horizon)  # [B, N, H]
        out = out.permute(0, 2, 1)          # [B, H, N]

        return out


# ============================================================================
# FACTORY
# ============================================================================

def build_stgnn(
        in_features: int = 1,
        gcn_hidden: int = 16,
        gcn_layers: int = 1,
        gru_hidden: int = 32,
        horizon: int = 6,
        dropout: float = 0.1
) -> STGNNModel:
    """
    Convenience builder that:
    - Loads adjacency + node count
    - Instantiates STGNNModel with sensible defaults
    """
    adj, num_nodes = load_adjacency_and_num_nodes()
    model = STGNNModel(
        num_nodes=num_nodes,
        in_features=in_features,
        gcn_hidden=gcn_hidden,
        gcn_layers=gcn_layers,
        gru_hidden=gru_hidden,
        horizon=horizon,
        dropout=dropout,
        adj=adj
    )
    return model


# ============================================================================
# QUICK SHAPE TEST (OPTIONAL)
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    model = build_stgnn().to(device)
    print(model)

    # Dummy batch to verify shape
    B = 4
    T_in = 24
    H_out = 6

    with NODE_MAP_PATH.open("r") as f:
        num_nodes = len(json.load(f))

    x_dummy = torch.randn(B, T_in, num_nodes, 1, device=device)
    y_hat = model(x_dummy)

    print(f"Input shape:  {x_dummy.shape}")
    print(f"Output shape: {y_hat.shape}")  # [B, H, N]
