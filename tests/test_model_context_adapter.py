import torch

from dymad.core import (
    FixedGraphSeries,
    GraphModelContext,
    GraphSeriesBatch,
    RegularModelContext,
    RegularSeries,
    RegularSeriesBatch,
)
from dymad.models.components import enc_graph_iden, enc_iden, zu_cat_smpl, zu_cat_smpl_graph


def test_regular_model_context_preserves_legacy_runtime_fields():
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

    assert context.batch_size == 2
    assert context.n_steps == (3, 3)
    assert torch.equal(
        context.initial_state_tensor(),
        torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
    )

    legacy = context.to_legacy_runtime()
    assert legacy.batch_size == 2
    assert legacy.n_steps == 3
    assert torch.equal(legacy.x[:, 0, :], context.initial_state_tensor())
    assert torch.equal(enc_iden(None, legacy.get_step(0)), context.initial_state_tensor())

    z = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    expected = torch.cat([z, legacy.get_step(0).u], dim=-1)
    assert torch.equal(zu_cat_smpl(z, legacy.get_step(0)), expected)


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

    legacy = context.to_legacy_runtime()
    step0 = legacy.get_step(0)
    assert legacy.batch_size == 2
    assert legacy._has_graph
    assert torch.equal(
        step0.xg,
        torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            ]
        ),
    )
    assert torch.equal(enc_graph_iden(None, step0), step0.xg)

    z = torch.tensor(
        [[[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]]
    )
    expected = torch.cat([z, step0.ug], dim=-1)
    assert torch.equal(zu_cat_smpl_graph(z, step0), expected)
