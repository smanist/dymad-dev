import sys
from types import ModuleType

import torch

import dymad.io.checkpoint as checkpoint_module
from dymad.io.checkpoint import (
    _prepare_visualize_model_input,
    _prune_visual_graph_to_output_paths,
    visualize_model,
)


class _FakeVisualGraph:
    def __init__(self) -> None:
        self.body = [
            "\t0 [label=<input-tensor> fillcolor=lightyellow]\n",
            "\t1 [label=<input-tensor> fillcolor=lightyellow]\n",
            "\t2 [label=<Sub> fillcolor=darkseagreen1]\n",
            "\t3 [label=<output-tensor> fillcolor=lightyellow]\n",
            "\t4 [label=<Other> fillcolor=darkseagreen1]\n",
            "\t0 -> 2\n",
            "\t1 -> 4\n",
            "\t2 -> 3\n",
        ]


class _FakeNode:
    def __init__(self, node_id: str, name: str) -> None:
        self.node_id = node_id
        self.name = name


class _FakeModelGraph:
    def __init__(self) -> None:
        input_a = _FakeNode("input_a", "input-tensor")
        input_b = _FakeNode("input_b", "input-tensor")
        output_subgraph = _FakeNode("output_subgraph", "Sub")
        output = _FakeNode("output", "output-tensor")
        unused_subgraph = _FakeNode("unused_subgraph", "Other")
        self.edge_list = [
            (input_a, output_subgraph),
            (input_b, unused_subgraph),
            (output_subgraph, output),
        ]
        self.id_dict = {
            "input_a": 0,
            "input_b": 1,
            "output_subgraph": 2,
            "output": 3,
            "unused_subgraph": 4,
        }
        self.visual_graph = _FakeVisualGraph()
        self.resize_called = False

    def resize_graph(self) -> None:
        self.resize_called = True


def test_prune_visual_graph_to_output_paths_drops_non_output_branches():
    graph = _FakeModelGraph()

    _prune_visual_graph_to_output_paths(graph)

    graph_source = "".join(graph.visual_graph.body)
    assert "Other" not in graph_source
    assert "\t1 [" not in graph_source
    assert "1 -> 4" not in graph_source
    assert "Sub" in graph_source
    assert "output-tensor" in graph_source
    assert "0 -> 2" in graph_source
    assert "2 -> 3" in graph_source
    assert graph.resize_called is True


def test_visualize_model_skips_output_path_pruning_when_show_all_paths(monkeypatch):
    graph = _FakeModelGraph()
    calls = {"prune": 0}

    def fake_draw_graph(*args, **kwargs):
        return graph

    def fake_prune(model_graph):
        calls["prune"] += 1
        _prune_visual_graph_to_output_paths(model_graph)

    fake_torchview = ModuleType("torchview")
    fake_torchview.draw_graph = fake_draw_graph  # type: ignore[attr-defined]
    monkeypatch.setattr(checkpoint_module, "_prune_visual_graph_to_output_paths", fake_prune)
    monkeypatch.setitem(sys.modules, "torchview", fake_torchview)

    visual_graph = visualize_model(
        model=object(),
        prd_func=lambda *args, **kwargs: {
            "t": torch.tensor([0.0]),
            "x": torch.tensor([[1.0]]),
            "u": None,
            "p": None,
            "ei": None,
            "ew": None,
            "ea": None,
        },
        ref_data={},
        show_all_paths=True,
    )

    assert visual_graph is graph.visual_graph
    assert calls["prune"] == 0
    assert "Other" in "".join(graph.visual_graph.body)


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
