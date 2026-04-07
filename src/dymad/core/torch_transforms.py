"""Torch-native tensor transforms for the module-first migration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from dymad.core.transform_module import TransformModule


class IdentityTransform(TransformModule):
    def fit(self, data: Sequence[torch.Tensor]) -> IdentityTransform:
        if data:
            self.input_dim = int(data[0].shape[-1])
            self.output_dim = self.input_dim
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data


class AddOneTransform(TransformModule):
    def fit(self, data: Sequence[torch.Tensor]) -> AddOneTransform:
        if data:
            self.input_dim = int(data[0].shape[-1])
            self.output_dim = self.input_dim + 1
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*data.shape[:-1], 1, dtype=data.dtype, device=data.device)
        return torch.cat((data, ones), dim=-1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data[..., :-1]


def _cross_features(blocks: Sequence[torch.Tensor]) -> torch.Tensor:
    result = blocks[0]
    for block in blocks[1:]:
        result = (result.unsqueeze(-1) * block.unsqueeze(-2)).reshape(*result.shape[:-1], -1)
    return result


def _linear_index(sizes: Sequence[int], picks: Sequence[int]) -> int:
    idx = 0
    stride = 1
    for size, pick in zip(reversed(sizes), reversed(picks), strict=False):
        idx += pick * stride
        stride *= size
    return idx


class LiftTransform(TransformModule):
    def __init__(self, fobs=None, finv=None, **kwargs) -> None:
        super().__init__()
        self.fobs = fobs
        self.finv = finv
        self.kwargs = kwargs
        self._feature_sizes: list[int] | None = None

    def fit(self, data: Sequence[torch.Tensor]) -> LiftTransform:
        if not data:
            return self
        self.input_dim = int(data[0].shape[-1])
        sample = self.forward(data[0])
        self.output_dim = int(sample.shape[-1])
        self._feature_sizes = self._infer_feature_sizes()
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.fobs == "poly":
            return self._forward_poly(data)
        if self.fobs == "mixed":
            return self._forward_mixed(data)
        raise ValueError(
            "LiftTransform currently supports native fobs='poly' or fobs='mixed' only."
        )

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        if self._feature_sizes is None:
            raise ValueError("LiftTransform parameters are not initialized. Call fit(...) first.")
        if self.fobs == "poly":
            return self._inverse_poly(data)
        if self.fobs == "mixed":
            return self._inverse_mixed(data)
        raise ValueError(
            "LiftTransform currently supports native inverse only for fobs='poly' or fobs='mixed'."
        )

    def _infer_feature_sizes(self) -> list[int]:
        if self.fobs == "poly":
            ks = list(self.kwargs["Ks"])
            if len(ks) != self.input_dim:
                raise ValueError("LiftTransform poly Ks must align with input dimension.")
            return ks
        if self.fobs == "mixed":
            opts = self.kwargs["opts"]
            sizes = [0 for _ in range(self.input_dim)]
            for index, kind, order in opts:
                if kind == "m":
                    sizes[index] = int(order)
                elif kind == "f":
                    sizes[index] = 2 * int(order) + 1
                elif kind == "p":
                    sizes[index[0]] = int(order[0])
                    sizes[index[1]] = 2 * int(order[1]) + 1
                else:
                    raise ValueError(f"Unknown mixed lift kind: {kind}")
            return sizes
        raise ValueError("Unsupported native lift configuration.")

    def _forward_poly(self, data: torch.Tensor) -> torch.Tensor:
        ks = list(self.kwargs["Ks"])
        blocks = [
            torch.stack([data[..., dim] ** order for order in range(int(ks[dim]))], dim=-1)
            for dim in range(data.shape[-1])
        ]
        return _cross_features(blocks)

    def _forward_mixed(self, data: torch.Tensor) -> torch.Tensor:
        opts = self.kwargs["opts"]
        feature_blocks: list[torch.Tensor | None] = [None for _ in range(data.shape[-1])]
        used = set()
        for index, kind, order in opts:
            if kind == "m":
                feature_blocks[index] = torch.stack(
                    [data[..., index] ** degree for degree in range(int(order))],
                    dim=-1,
                )
                used.add(index)
            elif kind == "f":
                value = data[..., index]
                parts = [torch.ones_like(value)]
                for degree in range(1, int(order) + 1):
                    parts.append(torch.cos(degree * value))
                    parts.append(torch.sin(degree * value))
                feature_blocks[index] = torch.stack(parts, dim=-1)
                used.add(index)
            elif kind == "p":
                first, second = index
                radius = torch.sqrt(data[..., first] ** 2 + data[..., second] ** 2)
                theta = torch.atan2(data[..., second], data[..., first])
                feature_blocks[first] = torch.stack(
                    [radius**degree for degree in range(int(order[0]))],
                    dim=-1,
                )
                parts = [torch.ones_like(theta)]
                for degree in range(1, int(order[1]) + 1):
                    parts.append(torch.cos(degree * theta))
                    parts.append(torch.sin(degree * theta))
                feature_blocks[second] = torch.stack(parts, dim=-1)
                used.update(index)
            else:
                raise ValueError(f"Unknown mixed lift kind: {kind}")
        if used != set(range(data.shape[-1])):
            raise ValueError(
                "LiftTransform mixed options must cover every input dimension exactly once."
            )
        return _cross_features(feature_blocks)  # type: ignore[arg-type]

    def _inverse_poly(self, data: torch.Tensor) -> torch.Tensor:
        assert self._feature_sizes is not None
        outputs = []
        for dim in range(len(self._feature_sizes)):
            picks = [0 for _ in self._feature_sizes]
            picks[dim] = 1
            outputs.append(data[..., _linear_index(self._feature_sizes, picks)])
        return torch.stack(outputs, dim=-1)

    def _inverse_mixed(self, data: torch.Tensor) -> torch.Tensor:
        assert self._feature_sizes is not None
        result = torch.zeros(
            *data.shape[:-1], len(self._feature_sizes), dtype=data.dtype, device=data.device
        )
        for index, kind, _order in self.kwargs["opts"]:
            if kind == "m":
                picks = [0 for _ in self._feature_sizes]
                picks[index] = 1
                result[..., index] = data[..., _linear_index(self._feature_sizes, picks)]
            elif kind == "f":
                cos_picks = [0 for _ in self._feature_sizes]
                sin_picks = [0 for _ in self._feature_sizes]
                cos_picks[index] = 1
                sin_picks[index] = 2
                result[..., index] = torch.atan2(
                    data[..., _linear_index(self._feature_sizes, sin_picks)],
                    data[..., _linear_index(self._feature_sizes, cos_picks)],
                )
            elif kind == "p":
                first, second = index
                r_picks = [0 for _ in self._feature_sizes]
                cos_picks = [0 for _ in self._feature_sizes]
                sin_picks = [0 for _ in self._feature_sizes]
                r_picks[first] = 1
                cos_picks[second] = 1
                sin_picks[second] = 2
                radius = data[..., _linear_index(self._feature_sizes, r_picks)]
                cos_theta = data[..., _linear_index(self._feature_sizes, cos_picks)]
                sin_theta = data[..., _linear_index(self._feature_sizes, sin_picks)]
                result[..., first] = radius * cos_theta
                result[..., second] = radius * sin_theta
            else:
                raise ValueError(f"Unknown mixed lift kind: {kind}")
        return result


class ScalerTransform(TransformModule):
    def __init__(self, mode: str = "01") -> None:
        super().__init__()
        self.mode = mode.lower()
        self.register_buffer("offset", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def fit(self, data: Sequence[torch.Tensor]) -> ScalerTransform:
        if not data:
            return self
        merged = torch.cat([item.reshape(-1, item.shape[-1]) for item in data], dim=0)
        features = merged.shape[-1]
        if self.mode == "01":
            offset = torch.amin(merged, dim=0)
            scale = torch.amax(merged, dim=0) - offset
        elif self.mode == "-11":
            offset = torch.zeros(features, dtype=merged.dtype, device=merged.device)
            scale = torch.amax(torch.abs(merged), dim=0)
        elif self.mode == "std":
            offset = torch.mean(merged, dim=0)
            scale = torch.std(merged, dim=0, unbiased=False)
        elif self.mode == "none":
            offset = torch.zeros(features, dtype=merged.dtype, device=merged.device)
            scale = torch.ones(features, dtype=merged.dtype, device=merged.device)
        else:
            raise ValueError(f"Unknown scaling mode: {self.mode}")
        scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
        self.offset = offset
        self.scale = scale
        self.input_dim = features
        self.output_dim = features
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.offset.numel() == 0 or self.scale.numel() == 0:
            raise ValueError("ScalerTransform parameters are not initialized. Call fit(...) first.")
        return (data - self.offset) / self.scale

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        if self.offset.numel() == 0 or self.scale.numel() == 0:
            raise ValueError("ScalerTransform parameters are not initialized. Call fit(...) first.")
        return data * self.scale + self.offset


class DelayEmbeddingTransform(TransformModule):
    def __init__(self, delay: int = 1) -> None:
        super().__init__(delay=delay)

    def fit(self, data: Sequence[torch.Tensor]) -> DelayEmbeddingTransform:
        if data:
            self.input_dim = int(data[0].shape[-1])
            self.output_dim = self.input_dim * (self.delay + 1)
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        seq_length = data.shape[0]
        if seq_length <= self.delay:
            raise ValueError(
                f"Sequence length ({seq_length}) must be greater than delay ({self.delay})."
            )
        length = seq_length - self.delay
        windows = [data[offset : length + offset] for offset in range(self.delay + 1)]
        return torch.cat(windows, dim=-1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        if self.input_dim is None:
            raise ValueError(
                "DelayEmbeddingTransform parameters are not initialized. Call fit(...) first."
            )
        head = data[..., : self.input_dim]
        if self.delay <= 0:
            return head
        tail = data[-1, ..., self.input_dim :]
        mid_dims = data.shape[1:-1]
        tail = tail.reshape(*mid_dims, self.delay, self.input_dim)
        perm = [len(mid_dims)] + list(range(len(mid_dims))) + [len(mid_dims) + 1]
        tail = tail.permute(*perm)
        return torch.cat((head, tail), dim=0)


def _resolve_svd_rank(singular_values: torch.Tensor, order, *, beta: float = 1.0) -> int:
    if isinstance(order, float):
        if order > 0:
            s2 = singular_values.square()
            ratio = torch.cumsum(s2, dim=0) / torch.sum(s2)
            idx = int(torch.argmax((ratio > order).to(torch.int64)).item())
            return max(1, idx)
        n = int(singular_values.numel())
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        mask = singular_values < omega * torch.median(singular_values)
        if not torch.any(mask):
            return n
        return max(1, int(torch.argmax(mask.to(torch.int64)).item()))
    if isinstance(order, int):
        if order >= 0:
            return max(1, min(int(order), int(singular_values.numel())))
        return max(1, int(singular_values.numel()) + int(order))
    if isinstance(order, str) and order.lower() == "full":
        return int(singular_values.numel())
    raise NotImplementedError(f"Undefined threshold for order={order}")


class SVDTransform(TransformModule):
    def __init__(self, order: int | float | str = 1.0, ifcen: bool = False) -> None:
        super().__init__(invertibility="approximate")
        self.order = order
        self.ifcen = bool(ifcen)
        self.register_buffer("projection", torch.empty(0))
        self.register_buffer("offset", torch.empty(0))

    def fit(self, data: Sequence[torch.Tensor]) -> SVDTransform:
        if not data:
            return self
        merged = torch.cat([item.reshape(-1, item.shape[-1]) for item in data], dim=0)
        self.input_dim = int(merged.shape[-1])
        if self.ifcen:
            offset = torch.mean(merged, dim=0)
            centered = merged - offset
        else:
            offset = torch.zeros(self.input_dim, dtype=merged.dtype, device=merged.device)
            centered = merged
        _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        beta = min(centered.shape) / max(centered.shape)
        rank = _resolve_svd_rank(singular_values, self.order, beta=float(beta))
        projection = vh[:rank].transpose(0, 1).contiguous()
        self.projection = projection
        self.offset = offset
        self.output_dim = int(projection.shape[1])
        self.invertibility = "exact" if self.output_dim == self.input_dim else "approximate"
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.projection.numel() == 0:
            raise ValueError("SVDTransform parameters are not initialized. Call fit(...) first.")
        return (data - self.offset).matmul(self.projection)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        if self.projection.numel() == 0:
            raise ValueError("SVDTransform parameters are not initialized. Call fit(...) first.")
        return data.matmul(self.projection.transpose(0, 1)) + self.offset


class AutoencoderTransform(TransformModule):
    def __init__(self, model, encoder, decoder) -> None:
        super().__init__(invertibility="approximate", supports_gradients="true")
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = model.n_total_state_features
        self.output_dim = model.latent_dimension
        self.dtype = model.dtype
        self.device = model.device

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.encoder(data.to(device=self.device, dtype=self.dtype))

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return self.decoder(data.to(device=self.device, dtype=self.dtype))


class ComposeTransform(TransformModule):
    def __init__(self, transforms: Sequence[TransformModule] | None = None) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(list(transforms or []))
        self._refresh_metadata()

    @property
    def NT(self) -> int:
        return len(self.transforms)

    def append(self, transform: TransformModule) -> None:
        self.transforms.append(transform)
        self._refresh_metadata()

    def fit(self, data: Sequence[torch.Tensor]) -> ComposeTransform:
        current = list(data)
        for stage in self.transforms:
            stage.fit(current)
            current = stage.transform_batch(current)
        if self.transforms:
            self.input_dim = self.transforms[0].input_dim
            self.output_dim = self.transforms[-1].output_dim
        self._refresh_metadata()
        return self

    def _proc_rng(self, rng: list[int] | None) -> list[int]:
        if rng is None:
            return [0, self.NT]
        if len(rng) != 2:
            raise ValueError("Range should be a two-element list [start, end].")
        start, end = int(rng[0]), int(rng[1])
        if not 0 <= start < end <= self.NT:
            raise ValueError(f"Range should be within [0, {self.NT}].")
        return [start, end]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward_range(data)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return self.inverse_range(data)

    def forward_range(self, data: torch.Tensor, rng: list[int] | None = None) -> torch.Tensor:
        start, end = self._proc_rng(rng)
        for stage in self.transforms[start:end]:
            data = stage(data)
        return data

    def inverse_range(self, data: torch.Tensor, rng: list[int] | None = None) -> torch.Tensor:
        start, end = self._proc_rng(rng)
        for stage in reversed(self.transforms[start:end]):
            data = stage.inverse(data)
        return data

    def _module_for_range(self, rng: list[int]) -> TransformModule:
        start, end = self._proc_rng(rng)
        if end - start == 1:
            return self.transforms[start]
        return ComposeTransform(list(self.transforms[start:end]))

    def _refresh_metadata(self) -> None:
        delayed = [stage for stage in self.transforms if stage.delay > 0]
        if len(delayed) > 1:
            raise ValueError("ComposeTransform supports at most one delayed stage.")
        self.delay = delayed[0].delay if delayed else 0
        if self.transforms:
            self.input_dim = self.transforms[0].input_dim
            self.output_dim = self.transforms[-1].output_dim
        self.invertibility = self._aggregate_invertibility()
        self.supports_gradients = self._aggregate_gradient_support()

    def _aggregate_invertibility(self) -> str:
        if any(stage.invertibility == "none" for stage in self.transforms):
            return "none"
        if any(stage.invertibility == "approximate" for stage in self.transforms):
            return "approximate"
        return "exact"

    def _aggregate_gradient_support(self) -> str:
        if any(stage.supports_gradients == "false" for stage in self.transforms):
            return "false"
        if any(stage.supports_gradients == "approximate" for stage in self.transforms):
            return "approximate"
        return "true"
