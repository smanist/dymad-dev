"""Library-backed vortex transform/mode analysis extracted from the script path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dymad.io import DataInterface

TRN_SVD = {"type": "svd", "ifcen": True, "order": 0.9999}
TRN_DMF = {
    "type": "dm",
    "edim": 3,
    "Knn": 15,
    "Kphi": 3,
    "inverse": "gmls",
    "order": 1,
    "mode": "full",
}


@dataclass(frozen=True)
class VortexModeAnalysisResult:
    output_path: str
    summary_path: str
    rel_dx_error: float
    rel_dz_error: float
    index: int
    nx: int
    ny: int


def compute_vortex_mode_analysis(
    *,
    config_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    index: int = 5,
    nx: int = 199,
    ny: int = 449,
) -> dict[str, np.ndarray | float | int]:
    train_data = np.load(train_dataset_path)
    test_data = np.load(test_dataset_path)
    t_train = train_data["t"]
    x_train = train_data["x"]
    x_test = test_data["x"]
    dt = float(t_train[1] - t_train[0])

    if len(x_test) < 3:
        raise ValueError("Need at least three test snapshots to estimate finite differences.")
    if not 0 < index < len(x_test) - 1:
        raise ValueError(f"index must be between 1 and {len(x_test) - 2}, got {index}")

    di = DataInterface(
        config_path=config_path,
        config_mod={
            "data": {"path": str(train_dataset_path)},
            "transform_x": [dict(TRN_SVD), dict(TRN_DMF)],
        },
    )
    z_train = di.encode(x_train)
    z_svd = di.encode(x_test, rng=[0, 1])
    z_dmf = di.encode(z_svd, rng=[1, 2])
    x_svd = di.decode(z_dmf, rng=[1, 2])
    x_rec = di.decode(x_svd, rng=[0, 1])

    ref = x_test[index].reshape(1, nx, ny)
    dx_ref = ((x_test[index + 1] - x_test[index - 1]) / (2 * dt)).reshape(1, nx, ny)
    dz_ref = (z_dmf[index + 1] - z_dmf[index - 1]) / (2 * dt)
    modes_backward = di.get_backward_modes(ref=z_dmf[index]).reshape(-1, nx, ny)
    modes_forward = di.get_forward_modes(ref=x_test[index]).reshape(-1, nx, ny)

    dx_est = np.sum(dz_ref[:, None, None] * modes_backward, axis=0, keepdims=True)
    dz_est = np.sum(dx_ref * modes_forward, axis=(1, 2))

    flat_forward = modes_forward.reshape(modes_forward.shape[0], -1)
    flat_backward = modes_backward.reshape(modes_backward.shape[0], -1)
    overlap_fb = flat_forward @ flat_backward.T
    overlap_ff = flat_forward @ flat_forward.T
    overlap_bb = flat_backward @ flat_backward.T

    rel_dx_error = float(np.linalg.norm(dx_est - dx_ref) / np.linalg.norm(dx_ref))
    rel_dz_error = float(np.linalg.norm(dz_est - dz_ref) / np.linalg.norm(dz_ref))

    return {
        "index": int(index),
        "dt": dt,
        "t_train": t_train,
        "x_train": x_train,
        "x_test": x_test,
        "z_train": z_train,
        "z_svd": z_svd,
        "z_dmf": z_dmf,
        "x_svd": x_svd,
        "x_rec": x_rec,
        "ref": ref,
        "dx_ref": dx_ref,
        "dz_ref": dz_ref,
        "modes_backward": modes_backward,
        "modes_forward": modes_forward,
        "dx_est": dx_est,
        "dz_est": dz_est,
        "overlap_fb": overlap_fb,
        "overlap_ff": overlap_ff,
        "overlap_bb": overlap_bb,
        "rel_dx_error": rel_dx_error,
        "rel_dz_error": rel_dz_error,
    }


def persist_vortex_mode_analysis(
    analysis: dict[str, np.ndarray | float | int],
    *,
    artifact_root: str | Path,
    stem: str = "vortex_transform_modes",
) -> VortexModeAnalysisResult:
    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{stem}.npz"
    summary_path = root / f"{stem}_summary.json"
    np.savez_compressed(output_path, **cast(Any, analysis))
    summary_payload: dict[str, Any] = {
        "workflow_key": "vortex_transform_modes",
        "index": int(analysis["index"]),
        "nx": int(np.asarray(analysis["ref"]).shape[-2]),
        "ny": int(np.asarray(analysis["ref"]).shape[-1]),
        "rel_dx_error": float(analysis["rel_dx_error"]),
        "rel_dz_error": float(analysis["rel_dz_error"]),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    return VortexModeAnalysisResult(
        output_path=str(output_path),
        summary_path=str(summary_path),
        rel_dx_error=float(analysis["rel_dx_error"]),
        rel_dz_error=float(analysis["rel_dz_error"]),
        index=int(analysis["index"]),
        nx=int(summary_payload["nx"]),
        ny=int(summary_payload["ny"]),
    )
