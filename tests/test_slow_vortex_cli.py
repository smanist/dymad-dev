from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.vortex.data import data_extract

from tests.slow_regression_utils import build_mpl_env

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "vortex" / "vor_proc_cli.py"


def test_vortex_mat_checksum_rejects_changed_data(monkeypatch, tmp_path: Path):
    expected = b"expected vortex fixture"
    mat_path = tmp_path / data_extract.MAT_FILENAME
    mat_path.write_bytes(expected)
    monkeypatch.setattr(data_extract, "MAT_SHA256", hashlib.sha256(expected).hexdigest())

    data_extract._verify_mat_checksum(mat_path)
    mat_path.write_bytes(b"changed vortex fixture")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        data_extract._verify_mat_checksum(mat_path)


@pytest.mark.extra_slow
def test_vortex_proc_cli_modes(tmp_path: Path):
    env = build_mpl_env(tmp_path)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["XDG_CACHE_HOME"] = str(cache_dir)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--workdir",
            str(tmp_path),
            "--data",
            "--no-plot",
            "--no-show",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    output_path = tmp_path / "vor_proc_modes.npz"
    assert output_path.exists()

    with np.load(output_path) as npz:
        modes_backward = npz["modes_backward"]
        modes_forward = npz["modes_forward"]
        rel_dx_error = float(npz["rel_dx_error"])
        rel_dz_error = float(npz["rel_dz_error"])

    assert modes_backward.shape == (3, 199, 449)
    assert modes_forward.shape == (3, 199, 449)
    assert np.isfinite(modes_backward).all()
    assert np.isfinite(modes_forward).all()
    assert rel_dx_error < 0.12
    assert rel_dz_error < 0.08
