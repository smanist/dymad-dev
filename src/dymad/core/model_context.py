"""Typed model-runtime context adapters built from typed series objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.series import RegularSeries, RegularSeriesBatch

if TYPE_CHECKING:
    from dymad.io.data import DynData


@dataclass(frozen=True)
class RegularModelContext:
    """Typed regular runtime context for prediction/model helper entrypoints."""

    batch: RegularSeriesBatch

    @classmethod
    def from_series(cls, series: RegularSeries) -> "RegularModelContext":
        return cls(RegularSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: RegularSeriesBatch) -> "RegularModelContext":
        return cls(batch)

    @property
    def batch_size(self) -> int:
        return len(self.batch)

    @property
    def n_steps(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.batch)

    def initial_state_tensor(self, *, squeeze_single: bool = False) -> torch.Tensor:
        state = torch.stack([item.state[0] for item in self.batch], dim=0)
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_legacy_runtime(self) -> "DynData":
        from dymad.io.data import DynData
        from dymad.io.series_adapter import DynDataAdapter

        payloads = [DynDataAdapter.from_regular_series(item) for item in self.batch]
        return DynData.collate(payloads)


@dataclass(frozen=True)
class GraphModelContext:
    """Typed graph runtime context for prediction/model helper entrypoints."""

    batch: GraphSeriesBatch

    @classmethod
    def from_series(cls, series: GraphSeries) -> "GraphModelContext":
        return cls(GraphSeriesBatch.collate([series]))

    @classmethod
    def from_batch(cls, batch: GraphSeriesBatch) -> "GraphModelContext":
        return cls(batch)

    @property
    def batch_size(self) -> int:
        return len(self.batch)

    @property
    def n_steps(self) -> tuple[int, ...]:
        return tuple(int(item.time.shape[0]) for item in self.batch)

    @property
    def n_nodes(self) -> tuple[int, ...]:
        return tuple(int(item.node_state.shape[1]) for item in self.batch)

    def initial_state_tensor(self, *, squeeze_single: bool = False) -> torch.Tensor:
        state = torch.stack([item.node_state[0].reshape(-1) for item in self.batch], dim=0)
        if squeeze_single and state.shape[0] == 1:
            return state[0]
        return state

    def to_legacy_runtime(self) -> "DynData":
        from dymad.io.data import DynData
        from dymad.io.series_adapter import DynDataAdapter

        payloads = [DynDataAdapter.from_graph_series(item) for item in self.batch]
        return DynData.collate(payloads)


def build_model_context(
    batch: RegularSeries | RegularSeriesBatch | GraphSeries | GraphSeriesBatch,
) -> RegularModelContext | GraphModelContext:
    """Build a typed model-runtime context from typed regular or graph series payloads."""

    if isinstance(batch, RegularSeries):
        return RegularModelContext.from_series(batch)
    if isinstance(batch, RegularSeriesBatch):
        return RegularModelContext.from_batch(batch)
    if isinstance(batch, GraphSeries):
        return GraphModelContext.from_series(batch)
    if isinstance(batch, GraphSeriesBatch):
        return GraphModelContext.from_batch(batch)
    raise TypeError(f"Unsupported model-context payload type: {type(batch)!r}")


ModelRuntimePayload: TypeAlias = "DynData | RegularModelContext | GraphModelContext"


def _expand_regular_context_for_prediction(
    context: RegularModelContext,
    *,
    batch_size: int,
    is_batch: bool,
) -> RegularModelContext:
    if not is_batch:
        if context.batch_size != 1:
            raise ValueError(
                f"Single mode: ws batch size must be 1. Got ws: {context.batch_size}"
            )
        return context

    if context.batch_size == 1 and batch_size > 1:
        return RegularModelContext.from_batch(
            RegularSeriesBatch.collate(context.batch[0] for _ in range(batch_size))
        )
    if context.batch_size == batch_size:
        return context
    raise ValueError(
        f"Batch mode: ws batch size must be 1 or match x0. Got ws: {context.batch_size}, x0: {batch_size}"
    )


def _expand_graph_context_for_prediction(
    context: GraphModelContext,
    *,
    batch_size: int,
    is_batch: bool,
) -> GraphModelContext:
    if not is_batch:
        if context.batch_size != 1:
            raise ValueError(
                f"Single mode: ws batch size must be 1. Got ws: {context.batch_size}"
            )
        return context

    if context.batch_size == 1 and batch_size > 1:
        return GraphModelContext.from_batch(
            GraphSeriesBatch.collate(context.batch[0] for _ in range(batch_size))
        )
    if context.batch_size == batch_size:
        return context
    raise ValueError(
        f"Batch mode: ws batch size must be 1 or match x0. Got ws: {context.batch_size}, x0: {batch_size}"
    )


def _context_from_legacy_runtime(payload: "DynData") -> RegularModelContext | GraphModelContext:
    from dymad.io.series_adapter import SeriesAdapter

    return build_model_context(SeriesAdapter.from_dyndata(payload))


def materialize_model_base_forward_payload(
    *,
    t: torch.Tensor | None,
    x: torch.Tensor | None,
    u: torch.Tensor | None,
    p: torch.Tensor | None,
    ei: tuple[torch.Tensor, torch.Tensor] | None,
    ew: tuple[torch.Tensor, torch.Tensor] | None,
    ea: tuple[torch.Tensor, torch.Tensor] | None,
) -> RegularModelContext | GraphModelContext:
    """Build the model-base forward payload through one explicit compatibility seam."""

    from dymad.io.data import DynData

    if x is None:
        raise ValueError("model_base.forward requires `x` to materialize runtime payload.")

    if ei is None:
        if x.ndim == 1:
            state = x.reshape(1, 1, -1)
        elif x.ndim == 2:
            state = x.unsqueeze(1)
        elif x.ndim == 3:
            state = x
        else:
            raise ValueError(f"Unsupported regular forward input shape for x: {tuple(x.shape)}")

        if t is None:
            time = None
        elif t.ndim == 0:
            time = t.reshape(1, 1)
        elif t.ndim == 1:
            time = t.reshape(-1, 1)
        elif t.ndim == 2:
            time = t if t.shape[-1] == 1 else t[:, :1]
        else:
            raise ValueError(f"Unsupported regular forward input shape for t: {tuple(t.shape)}")

        if u is None:
            control = None
        elif u.ndim == 1:
            control = u.reshape(1, 1, -1)
        elif u.ndim == 2:
            control = u.unsqueeze(1)
        elif u.ndim == 3:
            control = u
        else:
            raise ValueError(f"Unsupported regular forward input shape for u: {tuple(u.shape)}")

        if p is None:
            params = None
        elif p.ndim == 1:
            params = p.unsqueeze(0)
        elif p.ndim == 2:
            params = p
        else:
            raise ValueError(f"Unsupported regular forward input shape for p: {tuple(p.shape)}")

        runtime = DynData(t=time, x=state, u=control, p=params)
        if runtime.batch_size is None or runtime.batch_size == 1:
            context = _context_from_legacy_runtime(runtime)
            if not isinstance(context, RegularModelContext):
                raise TypeError("Expected regular context from non-graph forward payload.")
            return context

        items = []
        for idx in range(runtime.batch_size):
            sample = DynData(
                t=runtime.t[idx : idx + 1] if runtime.t is not None else None,
                x=runtime.x[idx : idx + 1] if runtime.x is not None else None,
                u=runtime.u[idx : idx + 1] if runtime.u is not None else None,
                p=runtime.p[idx : idx + 1] if runtime.p is not None else None,
            )
            sample_context = _context_from_legacy_runtime(sample)
            if not isinstance(sample_context, RegularModelContext):
                raise TypeError("Expected regular context while splitting regular forward payload.")
            items.append(sample_context.batch[0])
        return RegularModelContext.from_batch(RegularSeriesBatch.collate(items))

    if x.ndim == 1:
        state = x.reshape(1, 1, -1)
    elif x.ndim == 2:
        state = x.unsqueeze(0)
    elif x.ndim == 3:
        state = x
    else:
        raise ValueError(f"Unsupported graph forward input shape for x: {tuple(x.shape)}")

    legacy_runtime = DynData(
        t=t,
        x=state,
        u=u,
        p=p,
        ei=torch.nested.nested_tensor_from_jagged(*ei),
        ew=torch.nested.nested_tensor_from_jagged(*ew) if ew is not None else None,
        ea=torch.nested.nested_tensor_from_jagged(*ea) if ea is not None else None,
    )
    context = _context_from_legacy_runtime(legacy_runtime)
    if not isinstance(context, GraphModelContext):
        raise TypeError("Expected graph context from graph forward payload.")
    return context


def materialize_prediction_runtime(
    payload: ModelRuntimePayload | None,
    *,
    batch_size: int,
    is_batch: bool,
) -> "DynData":
    """Materialize a prediction runtime payload through the typed-context boundary."""

    from dymad.io.data import DynData

    if payload is None:
        return DynData()

    if isinstance(payload, DynData):
        if not is_batch:
            if payload.batch_size is not None and payload.batch_size != 1:
                raise ValueError(
                    f"Single mode: ws batch size must be 1. Got ws: {payload.batch_size}"
                )
            return payload
        if payload.batch_size == batch_size:
            return payload
        if payload.batch_size != 1:
            raise ValueError(
                f"Batch mode: ws batch size must be 1 or match x0. Got ws: {payload.batch_size}, x0: {batch_size}"
            )
        context = _context_from_legacy_runtime(payload)
        if isinstance(context, RegularModelContext):
            return _expand_regular_context_for_prediction(
                context,
                batch_size=batch_size,
                is_batch=True,
            ).to_legacy_runtime()
        return _expand_graph_context_for_prediction(
            context,
            batch_size=batch_size,
            is_batch=True,
        ).to_legacy_runtime()

    if isinstance(payload, RegularModelContext):
        return _expand_regular_context_for_prediction(
            payload,
            batch_size=batch_size,
            is_batch=is_batch,
        ).to_legacy_runtime()

    if isinstance(payload, GraphModelContext):
        return _expand_graph_context_for_prediction(
            payload,
            batch_size=batch_size,
            is_batch=is_batch,
        ).to_legacy_runtime()

    raise TypeError(f"Unsupported runtime payload type: {type(payload)!r}")
