import logging

import numpy as np
import torch

Batch = (
    list[list[torch.Tensor]]
    | list[torch.Tensor]
    | torch.Tensor
    | list[list[np.ndarray]]
    | list[np.ndarray]
    | np.ndarray
)

try:
    from torch_geometric.utils import dense_to_sparse
except ImportError:
    logging.warning(
        "torch_geometric is not installed. GNN-related functionality will be unavailable."
    )
    dense_to_sparse = None


def adj_to_edge(adj: Batch) -> tuple[Batch, Batch]:
    """
    Convert dense adjacency matrix to edge index and edge weights.

    The inputs can be an arbitrary (nested) list or array of adjacency matrices,
    The outputs will be lists of edge indices and edge weights of the same hierarchy.

    dense_to_sparse is not directly used, because it aggregates all graphs into one big graph.
    But in our framework, we want to keep them separate.

    Lastly, edge_index is transposed to shape (num_edges, 2) for downstream graph-series adapters.
    """
    if isinstance(adj, list):
        ei, ew = [], []
        for a in adj:
            _ei, _ew = adj_to_edge(a)
            ei.append(_ei)
            ew.append(_ew)
        return ei, ew

    assert isinstance(adj, (np.ndarray, torch.Tensor)), (
        "adj must be list, np.ndarray, or torch.Tensor"
    )
    if adj.ndim > 2:
        ei, ew = [], []
        for a in adj:
            _ei, _ew = adj_to_edge(a)
            ei.append(_ei)
            ew.append(_ew)
        return ei, ew

    adj = torch.tensor(adj)
    if dense_to_sparse is None:
        raise ImportError("torch_geometric is required for dense adjacency to sparse conversion.")
    edge_index, edge_weights = dense_to_sparse(adj)
    return edge_index.numpy().T, edge_weights.numpy()
