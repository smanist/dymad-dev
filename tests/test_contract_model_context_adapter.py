import torch

from dymad.core import (
    FixedGraphSeries,
    GraphModelContext,
    GraphSeriesBatch,
    RaggedRegularRuntime,
    RegularModelContext,
    RegularSeries,
    RegularSeriesBatch,
    UniformGraphRuntime,
    UniformLengthRegularSeriesBatch,
    UniformRegularRuntime,
)
from dymad.core.model_context import (
    materialize_model_base_forward_payload,
    materialize_prediction_runtime,
)
from dymad.models.components import enc_graph_iden, enc_iden, zu_cat_smpl, zu_cat_smpl_graph


def test_regular_model_context_preserves_runtime_fields():
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                control=torch.tensor([[0.1], [0.2], [0.3]]),
                params=torch.tensor([9.0]),
                meta={"name": "a"},
            ),
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
                control=torch.tensor([[1.1], [1.2], [1.3]]),
                params=torch.tensor([8.0]),
                meta={"name": "b"},
            ),
        ]
    )

    context = RegularModelContext.from_batch(batch)
    runtime = context.to_runtime()

    assert isinstance(batch, UniformLengthRegularSeriesBatch)
    assert isinstance(runtime, UniformRegularRuntime)
    assert context.batch_size == 2
    assert context.n_steps == (3, 3)
    assert torch.equal(
        context.initial_state_tensor(),
        torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
    )
    assert torch.equal(runtime.x[:, 0, :], context.initial_state_tensor())
    assert torch.equal(enc_iden(None, runtime.get_step(0)), context.initial_state_tensor())

    z = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    expected = torch.cat([z, runtime.get_step(0).u], dim=-1)
    assert torch.equal(zu_cat_smpl(z, runtime.get_step(0)), expected)


def test_graph_model_context_preserves_graph_helper_inputs():
    batch = GraphSeriesBatch.collate(
        [
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
                meta={"name": "g1"},
            ),
            FixedGraphSeries(
                time=torch.tensor([0.0, 1.0]),
                node_state=torch.tensor(
                    [
                        [[10.0, 20.0], [30.0, 40.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ]
                ),
                control=torch.tensor(
                    [
                        [[1.1], [1.2]],
                        [[1.3], [1.4]],
                    ]
                ),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_weight=torch.tensor([3.0, 4.0]),
                meta={"name": "g2"},
            ),
        ]
    )

    context = GraphModelContext.from_batch(batch)
    runtime = context.to_runtime()
    step0 = runtime.get_step(0)

    assert isinstance(runtime, UniformGraphRuntime)
    assert context.batch_size == 2
    assert context.n_steps == (2, 2)
    assert context.n_nodes == (2, 2)
    assert torch.equal(
        context.initial_state_tensor(),
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
            ]
        ),
    )
    assert torch.equal(
        step0.xg,
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ]
        ),
    )
    assert torch.equal(enc_graph_iden(None, step0), step0.xg)

    z = torch.tensor(
        [
            [[1.0, 1.5], [2.0, 2.5]],
            [[3.0, 3.5], [4.0, 4.5]],
        ]
    )
    expected = torch.cat([z, step0.ug], dim=-1)
    assert torch.equal(zu_cat_smpl_graph(z, step0), expected)


def test_materialize_prediction_runtime_expands_regular_context_batches():
    context = RegularModelContext.from_series(
        RegularSeries(
            time=torch.tensor([0.0, 1.0]),
            state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            control=torch.tensor([[0.1], [0.2]]),
            params=torch.tensor([9.0]),
        )
    )

    runtime = materialize_prediction_runtime(context, batch_size=3, is_batch=True)

    assert isinstance(runtime, UniformRegularRuntime)
    assert runtime.batch_size == 3
    assert torch.equal(runtime.x[0], runtime.x[1])
    assert torch.equal(runtime.u[0], runtime.u[2])


def test_materialize_prediction_runtime_expands_single_typed_runtime():
    runtime = UniformRegularRuntime(
        time=torch.tensor([[0.0, 1.0]]),
        state=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        control=torch.tensor([[[0.1], [0.2]]]),
    )

    expanded = materialize_prediction_runtime(runtime, batch_size=2, is_batch=True)

    assert expanded.batch_size == 2
    assert torch.equal(expanded.x[0], expanded.x[1])
    assert torch.equal(expanded.u[0], expanded.u[1])


def test_regular_model_context_uses_ragged_runtime_for_uneven_batches():
    batch = RegularSeriesBatch.collate(
        [
            RegularSeries(
                time=torch.tensor([0.0, 1.0]),
                state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
            RegularSeries(
                time=torch.tensor([0.0, 1.0, 2.0]),
                state=torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
            ),
        ]
    )

    context = RegularModelContext.from_batch(batch)
    runtime = context.to_runtime()

    assert isinstance(runtime, RaggedRegularRuntime)
    assert runtime.batch_size == 2
    assert runtime.step_lengths == (2, 3)


def test_materialize_model_base_forward_payload_regular_context():
    payload = materialize_model_base_forward_payload(
        t=torch.tensor([0.0]),
        x=torch.tensor([[1.0, 2.0]]),
        u=torch.tensor([[0.5]]),
        p=None,
        ei=None,
        ew=None,
        ea=None,
    )

    assert isinstance(payload, RegularModelContext)
    assert payload.batch_size == 1
    assert torch.equal(payload.initial_state_tensor(), torch.tensor([[1.0, 2.0]]))


def test_materialize_model_base_forward_payload_graph_context():
    edge_index = torch.nested.nested_tensor(
        [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)],
        layout=torch.jagged,
    )
    edge_weight = torch.nested.nested_tensor(
        [torch.tensor([1.0, 2.0])],
        layout=torch.jagged,
    )

    payload = materialize_model_base_forward_payload(
        t=torch.tensor([0.0]),
        x=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        u=torch.tensor([[0.1, 0.2]]),
        p=None,
        ei=(edge_index.values(), edge_index.offsets()),
        ew=(edge_weight.values(), edge_weight.offsets()),
        ea=None,
    )

    assert isinstance(payload, GraphModelContext)
    assert payload.n_nodes == (2,)
    assert torch.equal(
        payload.initial_state_tensor(),
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )


def test_materialize_model_base_forward_payload_accepts_dense_graph_edge_tensor():
    payload = materialize_model_base_forward_payload(
        t=torch.tensor([[0.0]]),
        x=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        u=torch.tensor([[[0.1, 0.2]]]),
        p=None,
        ei=torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long),
        ew=torch.tensor([[1.0, 2.0]]),
        ea=None,
    )

    assert isinstance(payload, GraphModelContext)
    assert payload.n_nodes == (2,)
    assert torch.equal(payload.initial_state_tensor(), torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
