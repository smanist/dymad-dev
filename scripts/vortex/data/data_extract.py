"""Download the DMD book cylinder data and extract the vorticity snapshots."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA_URL = "http://dmdbook.com/DATA.zip"
MAT_FILENAME = "CYLINDER_ALL.mat"
RAW_FILENAME = "raw.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where the archive, MAT file, and raw.npz are stored.",
    )
    return parser.parse_args()


def download_mat_file(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    mat_path = data_dir / MAT_FILENAME
    if mat_path.exists():
        return mat_path

    archive_path = data_dir / "DATA.zip"
    print(f"Downloading vortex data from {DATA_URL} to {archive_path}")
    urllib.request.urlretrieve(DATA_URL, archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                member
                for member in archive.infolist()
                if Path(member.filename).name == MAT_FILENAME
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one {MAT_FILENAME} entry in {archive_path}, found {len(matches)}"
                )
            with archive.open(matches[0]) as source, mat_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    finally:
        archive_path.unlink(missing_ok=True)
    print(f"Extracted vortex MAT data: {mat_path}")
    return mat_path


def extract_raw_data(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser().resolve()
    raw_path = data_dir / RAW_FILENAME
    if raw_path.exists():
        return raw_path

    mat_path = download_mat_file(data_dir)
    raw = loadmat(mat_path, variable_names=["VORTALL"])
    if "VORTALL" not in raw:
        raise KeyError(f"{mat_path} does not contain VORTALL")
    vorticity = np.moveaxis(raw["VORTALL"].reshape(449, 199, 151), (0, 1, 2), (2, 1, 0))
    np.savez_compressed(raw_path, vor=vorticity)
    print(f"Generated vortex NPZ data: {raw_path}")
    return raw_path


def main() -> int:
    extract_raw_data(parse_args().data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
