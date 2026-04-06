from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import torch

from dymad.core import FixedGraphSeries, GraphTrainerBatch, RegularSeries, RegularTrainerBatch
from dymad.models.prediction import predict_discrete_exp
from dymad.models.runtime_view import build_component_input_view
from dymad.training.batch_adapter import batch_to_runtime


class _RegularExpModel:
    def encoder(self, payload):
        return build_component_input_view(payload).state

    def dynamics(self, z, payload):
        return z + 0.01

    def decoder(self, z, payload):
        return z


class _GraphExpModel:
    def encoder(self, payload):
        return build_component_input_view(payload).graph_state

    def dynamics(self, z, payload):
        return z + 0.01

    def decoder(self, z, payload):
        return z


def _make_regular_batch(batch_size: int, n_steps: int, n_state: int, n_control: int):
    items = []
    base_time = torch.linspace(0.0, 1.0, n_steps)
    for _idx in range(batch_size):
        state = torch.randn(n_steps, n_state)
        control = torch.randn(n_steps, n_control) if n_control > 0 else None
        items.append(
            RegularSeries(
                time=base_time.clone(),
                state=state,
                control=control,
            )
        )
    return RegularTrainerBatch.collate_series(items)


def _make_graph_batch(batch_size: int, n_steps: int, n_nodes: int, n_state: int, n_control: int):
    edge_src = torch.arange(n_nodes, dtype=torch.long)
    edge_dst = torch.roll(edge_src, shifts=-1)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    items = []
    base_time = torch.linspace(0.0, 1.0, n_steps)
    for _ in range(batch_size):
        node_state = torch.randn(n_steps, n_nodes, n_state)
        control = torch.randn(n_steps, n_nodes, n_control) if n_control > 0 else None
        items.append(
            FixedGraphSeries(
                time=base_time.clone(),
                node_state=node_state,
                edge_index=edge_index,
                control=control,
            )
        )
    return GraphTrainerBatch.collate_series(items)


def _measure(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return {
        "mean_ms": 1e3 * statistics.mean(values),
        "median_ms": 1e3 * statistics.median(values),
        "min_ms": 1e3 * min(values),
        "max_ms": 1e3 * max(values),
    }


def _describe(stats):
    return (
        f"mean={stats['mean_ms']:.2f}ms "
        f"median={stats['median_ms']:.2f}ms "
        f"min={stats['min_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark typed runtime hot paths.")
    parser.add_argument("--case", choices=["lti", "ltg", "ltga"], required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.case == "lti":
        batch = _make_regular_batch(batch_size=32, n_steps=128, n_state=8, n_control=2)
        model = _RegularExpModel()
    elif args.case == "ltg":
        batch = _make_graph_batch(batch_size=16, n_steps=64, n_nodes=8, n_state=4, n_control=2)
        model = _GraphExpModel()
    else:
        batch = _make_graph_batch(batch_size=16, n_steps=64, n_nodes=8, n_state=4, n_control=0)
        model = _GraphExpModel()

    batch_dev = batch.to(device)
    runtime = batch_to_runtime(batch_dev)
    x0 = runtime.initial_state()
    ts = runtime.t

    cases = {
        "batch_to_runtime": lambda: batch_to_runtime(batch_dev),
        "truncate": lambda: batch_dev.truncate(min(runtime.n_steps, 32)),
        "window": lambda: batch_dev.window(min(runtime.n_steps, 16), 4),
        "predict_discrete_exp": lambda: predict_discrete_exp(model, x0, ts, runtime),
    }

    print(f"case={args.case} device={device} iters={args.iters} warmup={args.warmup}")
    for name, fn in cases.items():
        stats = _measure(fn, warmup=args.warmup, iters=args.iters)
        print(f"{name:24s} {_describe(stats)}")


if __name__ == "__main__":
    main()
