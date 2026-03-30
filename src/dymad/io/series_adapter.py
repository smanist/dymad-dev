"""Compatibility adapters between typed regular series and legacy DynData."""

from __future__ import annotations

from typing import Any

import torch

from dymad.core.series import RegularSeries
from dymad.io.data import DynData


class SeriesAdapter:
    """Build typed series objects from legacy-compatible payloads."""

    @staticmethod
    def from_regular_arrays(
        time: Any,
        state: Any,
        *,
        control: Any = None,
        target: Any = None,
        params: Any = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RegularSeries:
        return RegularSeries(
            time=torch.as_tensor(time, dtype=dtype, device=device),
            state=torch.as_tensor(state, dtype=dtype, device=device),
            control=torch.as_tensor(control, dtype=dtype, device=device) if control is not None else None,
            target=torch.as_tensor(target, dtype=dtype, device=device) if target is not None else None,
            params=torch.as_tensor(params, dtype=dtype, device=device) if params is not None else None,
            meta=dict(meta or {}),
        )

    @staticmethod
    def from_dyndata(data: DynData) -> RegularSeries:
        if data._has_graph:
            raise ValueError("graph DynData is not supported by the regular-series adapter")

        def _squeeze_batch(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.ndim == 0:
                return tensor
            if tensor.ndim >= 1 and tensor.shape[0] == 1:
                return tensor[0]
            raise ValueError(
                "DynData batch_size > 1 is not supported by SeriesAdapter.from_dyndata; "
                "adapt one trajectory at a time"
            )

        return RegularSeries(
            time=_squeeze_batch(data.t),
            state=_squeeze_batch(data.x),
            control=_squeeze_batch(data.u),
            target=_squeeze_batch(data.y),
            params=_squeeze_batch(data.p),
            meta={"source": "DynData"},
        )


class DynDataAdapter:
    """Adapt typed regular series back to the legacy runtime object."""

    @staticmethod
    def from_regular_series(series: RegularSeries) -> DynData:
        return DynData(
            t=series.time,
            x=series.state,
            y=series.target,
            u=series.control,
            p=series.params,
            meta=[dict(series.meta)] if series.meta else [],
        )
