"""Filesystem-backed persistence for facade-boundary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dymad.sako.snapshot import KoopmanWeightSnapshot, SpectralSnapshot
from dymad.store.object_store import (
    CheckpointRecord,
    ObjectNotFoundError,
    ObjectSummary,
    PredictionRequestRecord,
    SpectralSnapshotRecord,
)


class FilesystemArtifactStore:
    """Persist facade/store records under one artifact root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self._KIND_DIRS.values():
            self._kind_dir(directory).mkdir(parents=True, exist_ok=True)

    _KIND_DIRS = {
        "checkpoint": "checkpoints",
        "prediction_request": "prediction_requests",
        "spectral_snapshot": "spectral_snapshots",
    }

    def persist_checkpoint(self, record: CheckpointRecord) -> None:
        payload = {
            "handle": record.handle,
            "model_ref": record.model_ref,
            "checkpoint_path": record.checkpoint_path,
            "device": record.device,
        }
        self._write_json("checkpoint", record.handle, payload)

    def load_checkpoint(self, handle: str) -> CheckpointRecord:
        payload = self._read_json("checkpoint", handle)
        return CheckpointRecord(
            handle=payload["handle"],
            model_ref=payload["model_ref"],
            checkpoint_path=payload["checkpoint_path"],
            device=payload["device"],
        )

    def persist_prediction_request(self, record: PredictionRequestRecord) -> None:
        payload = {
            "handle": record.handle,
            "checkpoint_handle": record.checkpoint_handle,
            "horizon": record.horizon,
            "has_control": record.has_control,
            "has_graph": record.has_graph,
        }
        self._write_json("prediction_request", record.handle, payload)

    def load_prediction_request(self, handle: str) -> PredictionRequestRecord:
        payload = self._read_json("prediction_request", handle)
        return PredictionRequestRecord(
            handle=payload["handle"],
            checkpoint_handle=payload["checkpoint_handle"],
            horizon=int(payload["horizon"]),
            has_control=bool(payload["has_control"]),
            has_graph=bool(payload["has_graph"]),
        )

    def persist_spectral_snapshot(self, record: SpectralSnapshotRecord) -> None:
        snapshot = record.snapshot
        payload = {
            "handle": record.handle,
            "checkpoint_handle": record.checkpoint_handle,
            "model_class": snapshot.model_class,
            "checkpoint_path": snapshot.checkpoint_path,
            "input_dim": snapshot.input_dim,
            "obs_dim": snapshot.obs_dim,
            "sample_count": snapshot.sample_count,
            "metadata": snapshot.metadata,
            "koopman_mode": snapshot.koopman_weights.mode,
        }
        arrays: dict[str, np.ndarray] = {
            "encoded_p0": snapshot.encoded_p0,
            "encoded_p1": snapshot.encoded_p1,
        }
        if snapshot.koopman_weights.mode == "full":
            arrays["full_matrix"] = self._require_array(
                snapshot.koopman_weights.full_matrix,
                name="full_matrix",
            )
        else:
            arrays["left_factor"] = self._require_array(
                snapshot.koopman_weights.left_factor,
                name="left_factor",
            )
            arrays["right_factor"] = self._require_array(
                snapshot.koopman_weights.right_factor,
                name="right_factor",
            )
        self._write_json("spectral_snapshot", record.handle, payload)
        np.savez_compressed(self._npz_path(record.handle), **arrays)

    def load_spectral_snapshot(self, handle: str) -> SpectralSnapshotRecord:
        payload = self._read_json("spectral_snapshot", handle)
        try:
            with np.load(self._npz_path(handle), allow_pickle=False) as arrays:
                koopman_mode = payload["koopman_mode"]
                if koopman_mode == "full":
                    weights = KoopmanWeightSnapshot(
                        mode="full",
                        full_matrix=np.array(arrays["full_matrix"]),
                    )
                elif koopman_mode == "low_rank":
                    weights = KoopmanWeightSnapshot(
                        mode="low_rank",
                        left_factor=np.array(arrays["left_factor"]),
                        right_factor=np.array(arrays["right_factor"]),
                    )
                else:
                    raise ValueError(f"unsupported koopman_mode: {koopman_mode}")
                snapshot = SpectralSnapshot(
                    model_class=payload["model_class"],
                    checkpoint_path=payload["checkpoint_path"],
                    encoded_p0=np.array(arrays["encoded_p0"]),
                    encoded_p1=np.array(arrays["encoded_p1"]),
                    koopman_weights=weights,
                    input_dim=int(payload["input_dim"]),
                    obs_dim=int(payload["obs_dim"]),
                    sample_count=int(payload["sample_count"]),
                    metadata=dict(payload.get("metadata", {})),
                )
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"missing spectral snapshot payload for handle: {handle}") from exc
        return SpectralSnapshotRecord(
            handle=payload["handle"],
            checkpoint_handle=payload["checkpoint_handle"],
            snapshot=snapshot,
        )

    def summarize(self, handle: str) -> ObjectSummary:
        kind = self._kind_for_handle(handle)
        payload = self._read_json(kind, handle)
        if kind == "checkpoint":
            return ObjectSummary(
                handle=payload["handle"],
                kind="checkpoint",
                derived_from=None,
                preview=f"{payload['model_ref']} @ {payload['checkpoint_path']}",
            )
        if kind == "prediction_request":
            return ObjectSummary(
                handle=payload["handle"],
                kind="prediction_request",
                derived_from=payload["checkpoint_handle"],
                preview=(
                    f"horizon={payload['horizon']}, "
                    f"control={payload['has_control']}, graph={payload['has_graph']}"
                ),
            )
        return ObjectSummary(
            handle=payload["handle"],
            kind="spectral_snapshot",
            derived_from=payload["checkpoint_handle"],
            preview=f"samples={payload['sample_count']}, obs_dim={payload['obs_dim']}",
        )

    def list_object_summaries(self, kind: str | None = None) -> list[ObjectSummary]:
        if kind is not None and kind not in self._KIND_DIRS:
            raise ValueError(f"unsupported object kind: {kind}")
        kinds = [kind] if kind is not None else list(self._KIND_DIRS)
        summaries: list[ObjectSummary] = []
        for active_kind in kinds:
            for path in sorted(self._kind_dir(self._KIND_DIRS[active_kind]).glob("*.json")):
                summaries.append(self.summarize(path.stem))
        return summaries

    def _write_json(self, kind: str, handle: str, payload: dict[str, Any]) -> None:
        path = self._json_path(kind, handle)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _read_json(self, kind: str, handle: str) -> dict[str, Any]:
        path = self._json_path(kind, handle)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"unknown {kind} handle: {handle}") from exc

    def _json_path(self, kind: str, handle: str) -> Path:
        directory = self._KIND_DIRS[kind]
        return self._kind_dir(directory) / f"{handle}.json"

    def _npz_path(self, handle: str) -> Path:
        return self._kind_dir(self._KIND_DIRS["spectral_snapshot"]) / f"{handle}.npz"

    def _kind_dir(self, directory: str) -> Path:
        return self.root / directory

    @staticmethod
    def _kind_for_handle(handle: str) -> str:
        if handle.startswith("chk_"):
            return "checkpoint"
        if handle.startswith("pred_"):
            return "prediction_request"
        if handle.startswith("specsnap_"):
            return "spectral_snapshot"
        raise ObjectNotFoundError(f"unknown handle: {handle}")

    @staticmethod
    def _require_array(value: np.ndarray | None, *, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"spectral snapshot is missing {name}")
        return value
