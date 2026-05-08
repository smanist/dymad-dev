import torch

from dymad.io.checkpoint import _prepare_visualize_model_input


def test_prepare_visualize_model_input_reduces_batched_prediction_payload():
    input_data = {
        "t": torch.arange(10, dtype=torch.float32).reshape(2, 5),
        "x": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "u": torch.arange(30, dtype=torch.float32).reshape(2, 5, 3),
        "p": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "ei": torch.tensor(
            [
                [[0, 1], [1, 0], [1, 2], [2, 1]],
                [[0, 2], [2, 0], [2, 1], [1, 2]],
            ],
            dtype=torch.long,
        ),
        "ew": None,
        "ea": None,
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(prepared["t"], torch.tensor(0.0))
    assert torch.equal(prepared["x"], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prepared["u"], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(prepared["p"], torch.tensor([0.0, 1.0]))
    assert torch.equal(
        prepared["ei"],
        torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long),
    )


def test_prepare_visualize_model_input_coarsens_graph_runtime_payload():
    input_data = {
        "t": torch.arange(6, dtype=torch.float32).reshape(1, 6),
        "x": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "u": torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
        "p": None,
        "ei": torch.tensor(
            [
                [
                    [[0, 1], [1, 0], [1, 2], [2, 1]],
                    [[0, 1], [1, 0], [1, 2], [2, 1]],
                ]
            ],
            dtype=torch.long,
        ),
        "ew": torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]),
        "ea": torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2),
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(
        prepared["ei"],
        torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long),
    )
    assert torch.equal(prepared["ew"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(
        prepared["ea"],
        torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
            ]
        ),
    )


def test_prepare_visualize_model_input_preserves_single_feature_edge_attr_shape():
    input_data = {
        "t": torch.arange(6, dtype=torch.float32).reshape(1, 6),
        "x": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "u": torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),
        "p": None,
        "ei": torch.tensor(
            [
                [
                    [[0, 1], [1, 0], [1, 2], [2, 1]],
                    [[0, 1], [1, 0], [1, 2], [2, 1]],
                ]
            ],
            dtype=torch.long,
        ),
        "ew": torch.tensor([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]),
        "ea": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4, 1),
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(prepared["ew"], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(
        prepared["ea"],
        torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    )
    assert prepared["ea"].shape == (4, 1)


def test_prepare_visualize_model_input_reduces_explicit_param_batches():
    input_data = {
        "t": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "x": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "u": None,
        "p": torch.tensor([[5.0], [6.0]]),
        "ei": None,
        "ew": None,
        "ea": None,
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(prepared["x"], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prepared["p"], torch.tensor([5.0]))


def test_prepare_visualize_model_input_preserves_shared_param_vectors():
    input_data = {
        "t": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "x": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "u": None,
        "p": torch.tensor([5.0, 6.0]),
        "ei": None,
        "ew": None,
        "ea": None,
    }

    prepared = _prepare_visualize_model_input(input_data)

    assert torch.equal(prepared["x"], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prepared["p"], torch.tensor([5.0, 6.0]))
