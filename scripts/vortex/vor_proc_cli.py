import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from dymad.agent.exec.vortex_analysis import (
    compute_vortex_mode_analysis,
    persist_vortex_mode_analysis,
)
from dymad.io import DataInterface

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = BASE_DIR / "vor_model.yaml"
DATA_EXTRACT_SCRIPT = DATA_DIR / "data_extract.py"
NX = 199
NY = 449
DEFAULT_SPLIT_INDEX = 140
DEFAULT_INDEX = 5

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
DEFAULT_PLOTS = ("backward", "forward", "correlation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run vortex preprocessing and mode analysis.")
    parser.add_argument(
        "--data",
        action="store_true",
        help="Generate train/test NPZ files from the raw vortex dataset.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DATA_DIR / "raw.npz",
        help="Path to the raw vortex NPZ with key 'vor'.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=DEFAULT_SPLIT_INDEX,
        help="Training/test split index used when generating data.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Optional directory for generated data and saved mode outputs.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=DEFAULT_INDEX,
        help="Test-set index used for mode analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vor_proc_modes.npz"),
        help="Where to save the computed mode arrays and metrics.",
    )
    parser.add_argument(
        "--embedding", action="store_true", help="Plot the DM embedding coordinates."
    )
    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="Plot a reconstruction comparison for the selected step.",
    )
    parser.add_argument("--svd", action="store_true", help="Plot SVD-space backward modes.")
    parser.add_argument("--backward", action="store_true", help="Plot backward modes.")
    parser.add_argument("--forward", action="store_true", help="Plot forward modes.")
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Plot forward/backward mode correlation matrices.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip all plotting.")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
    return parser.parse_args()


def resolve_root(workdir: Path | None) -> Path:
    return BASE_DIR if workdir is None else workdir.resolve()


def resolve_output_path(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def ensure_raw_data(raw_data_path: Path) -> Path:
    raw_data_path = raw_data_path.expanduser().resolve()
    if raw_data_path.exists():
        return raw_data_path
    if raw_data_path != (DATA_DIR / "raw.npz").resolve():
        raise FileNotFoundError(f"Raw vortex data not found: {raw_data_path}")

    subprocess.run(
        [sys.executable, str(DATA_EXTRACT_SCRIPT), "--data-dir", str(DATA_DIR)],
        check=True,
        cwd=BASE_DIR,
    )
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Vortex data extraction did not create {raw_data_path}")
    return raw_data_path


def generate_data(root: Path, raw_data_path: Path, split_index: int) -> tuple[Path, Path]:
    raw_data_path = ensure_raw_data(raw_data_path)
    dat = np.load(raw_data_path)["vor"]
    nt, nx, ny = dat.shape
    if (nx, ny) != (NX, NY):
        raise ValueError(f"Unexpected vortex grid {(nx, ny)}; expected {(NX, NY)}")
    if not 0 < split_index < nt:
        raise ValueError(f"split_index must be between 1 and {nt - 1}, got {split_index}")

    ts = np.arange(nt)
    x_all = dat.reshape(nt, -1)

    t_train = ts[:split_index]
    x_train = x_all[:split_index]
    t_test = ts[split_index:]
    x_test = x_all[split_index:]

    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    cylinder_path = data_root / "cylinder.npz"
    test_path = data_root / "test.npz"
    np.savez_compressed(cylinder_path, x=x_train, t=t_train)
    np.savez_compressed(test_path, x=x_test, t=t_test)
    print(f"Generated data: {cylinder_path}")
    print(f"Generated data: {test_path}")
    return cylinder_path, test_path


def ensure_processed_data(root: Path, raw_data_path: Path, split_index: int) -> tuple[Path, Path]:
    cylinder_path = root / "data" / "cylinder.npz"
    test_path = root / "data" / "test.npz"
    if cylinder_path.exists() and test_path.exists():
        return cylinder_path, test_path
    return generate_data(root, raw_data_path, split_index)


def compute_analysis(
    cylinder_path: Path, test_path: Path, index: int
) -> dict[str, np.ndarray | float | int]:
    return compute_vortex_mode_analysis(
        config_path=str(CONFIG_PATH),
        train_dataset_path=str(cylinder_path),
        test_dataset_path=str(test_path),
        index=index,
        nx=NX,
        ny=NY,
    )


def save_analysis(analysis: dict[str, np.ndarray | float | int], output_path: Path) -> None:
    persisted = persist_vortex_mode_analysis(
        analysis,
        artifact_root=output_path.parent,
        stem=output_path.stem,
    )
    print(f"Saved mode outputs: {persisted.output_path}")


def selected_plot_sections(args) -> set[str]:
    selected = {
        name
        for name in ("embedding", "reconstruction", "svd", "backward", "forward", "correlation")
        if getattr(args, name)
    }
    return selected or set(DEFAULT_PLOTS)


def plot_analysis(
    analysis: dict[str, np.ndarray | float | int], sections: set[str], cylinder_path: Path
) -> None:
    import matplotlib.pyplot as plt

    from dymad.utils import compare_contour, plot_contour

    index = int(analysis["index"])
    ref = np.asarray(analysis["ref"])
    z_train = np.asarray(analysis["z_train"])
    x_rec = np.asarray(analysis["x_rec"])
    x_svd = np.asarray(analysis["x_svd"])
    x_test = np.asarray(analysis["x_test"])
    t_train = np.asarray(analysis["t_train"])
    dz_ref = np.asarray(analysis["dz_ref"])
    dx_ref = np.asarray(analysis["dx_ref"])
    dx_est = np.asarray(analysis["dx_est"])
    dz_est = np.asarray(analysis["dz_est"])
    modes_backward = np.asarray(analysis["modes_backward"])
    modes_forward = np.asarray(analysis["modes_forward"])
    overlap_fb = np.asarray(analysis["overlap_fb"])
    overlap_ff = np.asarray(analysis["overlap_ff"])
    overlap_bb = np.asarray(analysis["overlap_bb"])

    if "embedding" in sections:
        plt.figure()
        for i in range(3):
            plt.plot(t_train, z_train[:, i], label=f"DM coord {i + 1}")
        plt.xlabel("t")
        plt.ylabel("z")
        plt.legend()

    if "reconstruction" in sections:
        x_true = x_test[index].reshape(NX, NY)
        x_rec_step = x_rec[index].reshape(NX, NY)
        _, ax = compare_contour(x_true, x_rec_step, vmin=-4, vmax=4, figsize=(12, 2))
        for axis in ax:
            axis.set_axis_off()

    if "svd" in sections:
        di = DataInterface(
            config_path=str(CONFIG_PATH),
            config_mod={
                "data": {"path": str(cylinder_path)},
                "transform_x": [dict(TRN_SVD), dict(TRN_DMF)],
            },
        )
        modes_svd = di.get_backward_modes(ref=x_svd[index], rng=[0, 1]).reshape(-1, NX, NY)
        arrays = np.concatenate([ref, 200 * modes_svd[:5]], axis=0)
        labels = [f"step {index}"] + [f"mode {i + 1}" for i in range(5)]
        _, ax = plot_contour(
            arrays, figsize=(12, 4), colorbar=True, label=labels, grid=(2, 3), mode="contourf"
        )
        for axis in ax.flatten():
            axis.set_axis_off()

    if "backward" in sections:
        arrays = np.concatenate([ref, modes_backward], axis=0)
        labels = [f"step {index}"] + [f"mode {i + 1}" for i in range(modes_backward.shape[0])]
        _, ax = plot_contour(
            arrays,
            vmin=-4,
            vmax=4,
            figsize=(8, 4),
            colorbar=True,
            label=labels,
            grid=(2, 2),
            mode="contourf",
        )
        for axis in ax.flatten():
            axis.set_axis_off()

        _, ax = compare_contour(dx_ref[0], dx_est[0], figsize=(12, 2))
        for axis in ax:
            axis.set_axis_off()

    if "forward" in sections:
        arrays = np.concatenate([ref, 40000 * modes_forward], axis=0)
        labels = [f"step {index}"] + [f"mode {i + 1}" for i in range(modes_forward.shape[0])]
        _, ax = plot_contour(
            arrays,
            vmin=-4,
            vmax=4,
            figsize=(8, 4),
            colorbar=True,
            label=labels,
            grid=(2, 2),
            mode="contourf",
        )
        for axis in ax.flatten():
            axis.set_axis_off()

        plt.figure()
        labels = [str(i + 1) for i in range(len(dz_est))]
        values = np.stack([dz_est, dz_ref], axis=1)
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, values[:, 0], width, label="Estimate", color="blue")
        plt.bar(x + width / 2, values[:, 1], width, label="Finite Diff.", color="orange")
        plt.xticks(x, labels)
        plt.legend()
        plt.ylabel("Rate")

    if "correlation" in sections:
        _, ax = plt.subplots(1, 3, figsize=(9, 3))
        im0 = ax[0].imshow(overlap_fb, vmin=-1, vmax=1, cmap="bwr")
        ax[0].set_title("dz/dx * dx/dz")
        plt.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(overlap_ff, cmap="bwr")
        ax[1].set_title("dz/dx * (dz/dx)^T")
        plt.colorbar(im1, ax=ax[1])

        im2 = ax[2].imshow(overlap_bb, cmap="bwr")
        ax[2].set_title("(dx/dz)^T * dx/dz")
        plt.colorbar(im2, ax=ax[2])


def main():
    args = parse_args()
    root = resolve_root(args.workdir)
    output_path = resolve_output_path(root, args.output)

    if args.data:
        generate_data(root, args.raw_data.resolve(), args.split_index)

    cylinder_path, test_path = ensure_processed_data(
        root, args.raw_data.resolve(), args.split_index
    )
    analysis = compute_analysis(cylinder_path, test_path, args.index)
    save_analysis(analysis, output_path)

    if not args.no_plot:
        plot_analysis(analysis, selected_plot_sections(args), cylinder_path)
    if not args.no_plot and not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
