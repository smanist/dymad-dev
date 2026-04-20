import torch

from dymad.io.checkpoint import _prepare_visualize_model_input


def test_prepare_visualize_model_input_normalizes_graph_prediction_payload():
    input_data = {
        "t": torch.arange(5, dtype=torch.float32).reshape(1, 5),
        "x": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "u": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "p": None,
        "ei": torch.tensor(
            [[[0, 1], [1, 0], [1, 2], [2, 1]]],
            dtype=torch.long,
        ),
        "ew": None,
        "ea": None,
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(prepared["t"], torch.tensor([[0.0]]))
    assert torch.equal(prepared["u"], torch.tensor([[[0.0, 1.0, 2.0]]]))
    assert torch.equal(prepared["ei"], input_data["ei"])
