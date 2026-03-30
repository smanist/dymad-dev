"""Torch-native tensor transforms for the module-first migration."""

from __future__ import annotations

from typing import Sequence

import torch

from dymad.core.transform_module import TransformModule


class IdentityTransform(TransformModule):
    def fit(self, data: Sequence[torch.Tensor]) -> "IdentityTransform":
        if data:
            self.input_dim = int(data[0].shape[-1])
            self.output_dim = self.input_dim
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data


class AddOneTransform(TransformModule):
    def fit(self, data: Sequence[torch.Tensor]) -> "AddOneTransform":
        if data:
            self.input_dim = int(data[0].shape[-1])
            self.output_dim = self.input_dim + 1
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*data.shape[:-1], 1, dtype=data.dtype, device=data.device)
        return torch.cat((data, ones), dim=-1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data[..., :-1]


class ScalerTransform(TransformModule):
    def __init__(self, mode: str = "01") -> None:
        super().__init__()
        self.mode = mode.lower()
        self.register_buffer("offset", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def fit(self, data: Sequence[torch.Tensor]) -> "ScalerTransform":
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

    def fit(self, data: Sequence[torch.Tensor]) -> "DelayEmbeddingTransform":
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
            raise ValueError("DelayEmbeddingTransform parameters are not initialized. Call fit(...) first.")
        head = data[..., : self.input_dim]
        if self.delay <= 0:
            return head
        tail = data[-1, ..., self.input_dim :]
        mid_dims = data.shape[1:-1]
        tail = tail.reshape(*mid_dims, self.delay, self.input_dim)
        perm = [len(mid_dims)] + list(range(len(mid_dims))) + [len(mid_dims) + 1]
        tail = tail.permute(*perm)
        return torch.cat((head, tail), dim=0)


class ComposeTransform(TransformModule):
    def __init__(self, transforms: Sequence[TransformModule]) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(list(transforms))
        delayed = [stage for stage in self.transforms if stage.delay > 0]
        if len(delayed) > 1:
            raise ValueError("ComposeTransform supports at most one delayed stage.")
        self.delay = delayed[0].delay if delayed else 0

    def fit(self, data: Sequence[torch.Tensor]) -> "ComposeTransform":
        current = list(data)
        for stage in self.transforms:
            stage.fit(current)
            current = stage.transform_batch(current)
        if self.transforms:
            self.input_dim = self.transforms[0].input_dim
            self.output_dim = self.transforms[-1].output_dim
        return self

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for stage in self.transforms:
            data = stage(data)
        return data

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        for stage in reversed(self.transforms):
            data = stage.inverse(data)
        return data
