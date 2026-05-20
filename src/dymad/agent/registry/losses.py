"""Loss registry accessors for agent-facing surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from dymad.losses import LOSS_MAP


@dataclass(frozen=True)
class LossCapability:
    key: str
    name: str
    implementation: str


def list_loss_capabilities() -> tuple[LossCapability, ...]:
    """List stable loss keys backed by the runtime loss map."""

    capabilities: list[LossCapability] = []
    for key, loss_cls in sorted(LOSS_MAP.items()):
        capabilities.append(
            LossCapability(
                key=key,
                name=getattr(loss_cls, "__name__", str(loss_cls)),
                implementation=f"{loss_cls.__module__}:{getattr(loss_cls, '__qualname__', key)}",
            )
        )
    return tuple(capabilities)
