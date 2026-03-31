import torch
import torch.nn as nn

from dymad.core import FixedGraphSeries, GraphModelContext, RegularModelContext, RegularSeries
from dymad.models.components import (
    enc_graph_ctrl,
    enc_smpl_ctrl,
    zu_blin_with_const,
    zu_blin_with_const_graph,
)


class EchoRegular(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return value


class EchoGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_args = None

    def forward(self, value, edge_index, edge_weight, edge_attr):
        self.last_args = (value, edge_index, edge_weight, edge_attr)
        return value.reshape(value.shape[0], -1)


def test_regular_component_helpers_accept_regular_model_context():
    context = RegularModelContext.from_series(
        RegularSeries(
            time=torch.tensor([0.0, 1.0]),
            state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            control=torch.tensor([[0.5], [0.6]]),
        )
    )
    encoder = EchoRegular()

    encoded = enc_smpl_ctrl(encoder, context)
    bilinear = zu_blin_with_const(torch.tensor([[2.0, 3.0]]), context)

    assert torch.equal(encoded, torch.tensor([[1.0, 2.0, 0.5]]))
    assert torch.equal(encoder.last_input, encoded)
    assert torch.equal(bilinear, torch.tensor([[2.0, 3.0, 1.0, 1.5, 0.5]]))


def test_graph_component_helpers_accept_graph_model_context():
    context = GraphModelContext.from_series(
        FixedGraphSeries(
            time=torch.tensor([0.0, 1.0]),
            node_state=torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            control=torch.tensor(
                [
                    [[0.1], [0.2]],
                    [[0.3], [0.4]],
                ]
            ),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_weight=torch.tensor([1.0, 2.0]),
        )
    )
    encoder = EchoGraph()

    encoded = enc_graph_ctrl(encoder, context)
    bilinear = zu_blin_with_const_graph(
        torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]),
        context,
    )

    assert encoded.shape == (1, 2, 3)
    value, edge_index, edge_weight, edge_attr = encoder.last_args
    assert value.shape == (1, 2, 3)
    assert edge_index is not None
    assert edge_weight is not None
    assert edge_attr is None
    assert torch.equal(
        bilinear,
        torch.tensor(
            [
                [
                    [2.0, 3.0, 0.2, 0.3, 0.1],
                    [4.0, 5.0, 0.8, 1.0, 0.2],
                ]
            ]
        ),
    )
