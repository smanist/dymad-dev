import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

from dymad.utils import adj_to_edge


def _d2s(adj):
    adj = torch.tensor(adj)
    edge_index, edge_weights = dense_to_sparse(adj)
    return edge_index.numpy().T, edge_weights.numpy()


def cmp_graph(e1, e2):
    """Not using recursion for explicitness."""
    if isinstance(e1[0], np.ndarray):
        assert np.array_equal(e1[0], e2[0]), "Single: Edge indices are not equal."
        assert np.array_equal(e1[1], e2[1]), "Single: Edge weights are not equal."
    elif isinstance(e1[0], list):
        if isinstance(e1[0][0], np.ndarray):
            assert len(e1[0]) == len(e2[0]), "Graph lists have different lengths in indices."
            assert len(e1[1]) == len(e2[1]), "Graph lists have different lengths in weights."
            for i in range(len(e1[0])):
                assert np.array_equal(e1[0][i], e2[0][i]), "List: Edge indices are not equal."
                assert np.array_equal(e1[1][i], e2[1][i]), "List: Edge weights are not equal."
        else:
            assert len(e1[0]) == len(e2[0]), "Nested graph lists have different lengths in indices."
            assert len(e1[1]) == len(e2[1]), "Nested graph lists have different lengths in weights."
            for i in range(len(e1[0])):
                assert len(e1[0][i]) == len(e2[0][i]), (
                    "Nested graph sub-lists have different lengths in indices."
                )
                assert len(e1[1][i]) == len(e2[1][i]), (
                    "Nested graph sub-lists have different lengths in weights."
                )
                for j in range(len(e1[0][i])):
                    assert np.array_equal(e1[0][i][j], e2[0][i][j]), (
                        "Nested List: Edge indices are not equal."
                    )
                    assert np.array_equal(e1[1][i][j], e2[1][i][j]), (
                        "Nested List: Edge weights are not equal."
                    )
    else:
        raise ValueError("Unsupported graph structure.")


A1 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
A2 = np.array([[0, 1, 1], [3.0, 0, 1.2], [0, 1, 0]])
e1 = _d2s(A1)
e2 = _d2s(A2)


def test_a2e_single():
    ew = adj_to_edge(A2)
    cmp_graph(ew, e2)


def test_a2e_list():
    adj_list = [A1, A2]
    ew = adj_to_edge(adj_list)
    expected = ([e1[0], e2[0]], [e1[1], e2[1]])
    cmp_graph(ew, expected)


def test_a2e_nested():
    adj_list = [[A1, A2], [A1]]
    ew = adj_to_edge(adj_list)
    expected = ([[e1[0], e2[0]], [e1[0]]], [[e1[1], e2[1]], [e1[1]]])
    cmp_graph(ew, expected)
