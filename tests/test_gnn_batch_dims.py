import torch

from dymad.modules.gnn import GNN


def test_gnn_accepts_graph_sequences_with_multiple_leading_batch_dims() -> None:
    net = GNN(
        input_dim=2,
        hidden_dim=4,
        output_dim=3,
        n_layers=1,
        gcl="sage",
        activation="none",
    )

    x = torch.randn(2, 3, 2, 2)
    edge_index = (
        torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        )
        .reshape(1, 1, 2, 2)
        .expand(2, 3, 2, 2)
    )
    edge_weight = torch.ones(2, 3, 2)

    out = net(x, edge_index, edge_weight, None)

    assert out.shape == (2, 3, 6)
