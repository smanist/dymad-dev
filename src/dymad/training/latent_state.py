from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch


class LatentState(ABC):
    """Abstract latent representation exchanged across training phases."""

    kind: str

    @abstractmethod
    def point(self) -> torch.Tensor:
        """Return a canonical point estimate of the latent trajectory."""
        raise NotImplementedError

    @abstractmethod
    def moments(self) -> Dict[str, torch.Tensor]:
        """Return statistical moments of the latent state."""
        raise NotImplementedError

    @abstractmethod
    def diagnostic_info(self) -> Dict[str, Any]:
        """Return diagnostic metadata."""
        raise NotImplementedError

    @abstractmethod
    def add_diagnostic(self, key: str, value: Any) -> None:
        """Attach a diagnostic entry."""
        raise NotImplementedError

    @abstractmethod
    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize to a checkpoint payload."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "LatentState":
        """Deserialize from checkpoint payload."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device=None, dtype=None) -> "LatentState":
        """Move latent tensors to target device/dtype."""
        raise NotImplementedError

    @abstractmethod
    def detach(self) -> "LatentState":
        """Detach latent tensors from autograd graph."""
        raise NotImplementedError


@dataclass
class MapLatent(LatentState):
    z_map: torch.Tensor
    diag: Dict[str, Any] = field(default_factory=dict)
    kind: str = "map"

    def point(self) -> torch.Tensor:
        return self.z_map

    def moments(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.z_map}

    def diagnostic_info(self) -> Dict[str, Any]:
        return dict(self.diag)

    def add_diagnostic(self, key: str, value: Any) -> None:
        self.diag[key] = value

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "z_map": self.z_map.detach().cpu(),
            "diag": dict(self.diag),
        }

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "MapLatent":
        return cls(
            z_map=ckpt["z_map"],
            diag=dict(ckpt.get("diag", {})),
        )

    def to(self, device=None, dtype=None) -> "MapLatent":
        if self.z_map is not None:
            self.z_map = self.z_map.to(device=device, dtype=dtype)
        return self

    def detach(self) -> "MapLatent":
        if self.z_map is not None:
            self.z_map = self.z_map.detach()
        return self


@dataclass
class GaussianDiagLatent(LatentState):
    mean: torch.Tensor
    var: torch.Tensor
    diag: Dict[str, Any] = field(default_factory=dict)
    kind: str = "gaussian_diag"

    def point(self) -> torch.Tensor:
        return self.mean

    def moments(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var}

    def diagnostic_info(self) -> Dict[str, Any]:
        return dict(self.diag)

    def add_diagnostic(self, key: str, value: Any) -> None:
        self.diag[key] = value

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "diag": dict(self.diag),
        }

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]) -> "GaussianDiagLatent":
        return cls(
            mean=ckpt["mean"],
            var=ckpt["var"],
            diag=dict(ckpt.get("diag", {})),
        )

    def to(self, device=None, dtype=None) -> "GaussianDiagLatent":
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.var = self.var.to(device=device, dtype=dtype)
        return self

    def detach(self) -> "GaussianDiagLatent":
        self.mean = self.mean.detach()
        self.var = self.var.detach()
        return self


LATENT_REGISTRY: Dict[str, Type[LatentState]] = {
    "map": MapLatent,
    "gaussian_diag": GaussianDiagLatent,
}


def latent_from_checkpoint(ckpt: Dict[str, Any]) -> LatentState:
    kind = ckpt.get("kind", None)
    if kind not in LATENT_REGISTRY:
        raise ValueError(f"Unknown latent state kind: {kind}")
    return LATENT_REGISTRY[kind].from_checkpoint(ckpt)
