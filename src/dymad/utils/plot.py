import logging
import os
from typing import Any, cast

import imageio
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

PALETTE = ["#000000", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
LINESTY = ["-", "--", "-.", ":", ".-"]

# Disable logging for matplotlib to avoid clutter in DEBUG mode
plt_logger = logging.getLogger("matplotlib")
plt_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def plot_trajectory(
    traj,
    ts,
    model_name=None,
    us=None,
    axes=None,
    labels=None,
    ifclose=True,
    prefix=".",
    xidx=None,
    uidx=None,
    grid=None,
    xscl=None,
    uscl=None,
    cmp_err=True,
):
    """
    Plot trajectories with optional control inputs and save the figure.

    Args:
        traj (np.ndarray): Trajectory data, shape (n_steps, n_features) or (n_traj, n_steps, n_features)
        ts (np.ndarray): Time points corresponding to the trajectory data, shape (n_steps,)
        model_name (str, optional): Name of the model for the plot title and filename
        us (np.ndarray, optional): Control inputs, shape (n_steps, n_controls)
        axes (list, optional): List of axes to plot on. If None, creates new subplots
        labels (list, optional): Labels for each trajectory, length must match number of trajectories
        ifclose (bool): Whether to close the plot after saving
        prefix (str): Directory prefix for saving the plot
        xidx (list, optional): Indices of state features to plot
        uidx (list, optional): Indices of control inputs to plot
        grid (tuple, optional): Tuple (n_rows, n_cols) for subplot layout
        xscl (str, optional): Scaling mode for state features ('01', '-11', 'std', or 'none')
        uscl (str, optional): Scaling mode for control inputs ('01', '-11', 'std', or 'none')
        cmp_err (bool): Whether to compute and display RMSE between trajectories

    Returns:
        axes (list): List of axes used for plotting
    """
    if traj.ndim == 2:
        traj = np.array([traj])

    Ntrj = len(traj)
    if labels is None:
        labels = [None] * Ntrj
    else:
        assert Ntrj == len(labels), "Number of trajectories must match number of labels"

    # Plot the first trajectory and create the axes
    _, ax = plot_one_trajectory(
        traj[0],
        ts,
        idx=0,
        us=us,
        axes=axes,
        label=labels[0],
        xidx=xidx,
        uidx=uidx,
        grid=grid,
        xscl=xscl,
        uscl=uscl,
    )

    if Ntrj > 1:
        # Add additional trajectories to the same axes
        for i in range(1, Ntrj):
            lbl = labels[i]
            if labels[i] is not None:
                if cmp_err:
                    rmse = np.linalg.norm(traj[0] - traj[i]) / (traj[0].shape[0] - 1) ** 0.5
                    lbl = cast(str, labels[i]) + f" rmse: {rmse:4.3e}"
            plot_one_trajectory(
                traj[i],
                ts,
                idx=i,
                us=None,
                axes=ax,
                label=lbl,
                xidx=xidx,
                uidx=uidx,
                grid=grid,
                xscl=xscl,
                uscl=uscl,
            )

    # Adjust layout and save
    plt.tight_layout()
    if model_name is not None:
        if prefix != ".":
            os.makedirs(prefix, exist_ok=True)
        plt.savefig(
            f"{prefix}/{model_name}_prediction.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    if ifclose:
        plt.close()

    return ax


def plot_multi_trajs(
    traj,
    ts,
    model_name,
    us=None,
    labels=None,
    ifclose=True,
    prefix=".",
    xidx=None,
    uidx=None,
    grid=None,
    xscl=None,
    uscl=None,
):
    """
    Multi-trajectory version of plot_trajectory - comparison of batches of trajectories.
    """
    if traj.ndim == 3:
        traj = np.array([traj])

    Ntrj = len(traj[0])
    if labels is None:
        labels = [f"Run {i + 1}" for i in range(len(traj))]
    assert len(traj) == len(labels), "Number of trajectories must match number of labels"
    _us = [None] * Ntrj if us is None else us

    # Update labels to include RMSE
    for i in range(1, len(traj)):
        rmse = np.sqrt(np.mean((traj[0] - traj[i]) ** 2))
        labels[i] = f"{labels[i]} rmse: {rmse:4.3e}"

    # Plot the first trajectory and create the axes
    ax = plot_trajectory(
        np.array([_t[0] for _t in traj]),
        ts,
        model_name=None,
        us=_us[0],
        axes=None,
        labels=labels,
        ifclose=False,
        xidx=xidx,
        uidx=uidx,
        grid=grid,
        xscl=xscl,
        uscl=uscl,
        cmp_err=False,
    )

    if Ntrj > 1:
        # Add additional trajectories to the same axes
        for i in range(1, Ntrj):
            ax = plot_trajectory(
                np.array([_t[i] for _t in traj]),
                ts,
                model_name=None,
                us=_us[i],
                axes=ax,
                labels=None,
                ifclose=False,
                xidx=xidx,
                uidx=uidx,
                grid=grid,
                xscl=xscl,
                uscl=uscl,
            )

    # Adjust layout and save
    plt.tight_layout()
    if prefix != ".":
        os.makedirs(prefix, exist_ok=True)
    plt.savefig(
        f"{prefix}/{model_name}_prediction.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    if ifclose:
        plt.close()


def _scale_axes(ax, data, scl):
    if scl is None:
        return
    if scl == "01":
        ax.set_ylim([-0.1, 1.1])  # [0,1] range with small buffer
    elif scl == "-11":
        ax.set_ylim([-1.2, 1.2])  # [-1,1] range with buffer
    elif scl == "std":
        ax.set_ylim([-3, 3])  # ±3 std devs for standardized data
    else:  # mode=none
        ymx = np.max(data)
        ymn = np.min(data)
        ax.set_ylim([ymn - 0.1 * abs(ymn), ymx + 0.1 * abs(ymx)])  # Use data range with buffer


def plot_one_trajectory(
    traj,
    ts,
    idx=0,
    us=None,
    axes=None,
    label=None,
    xidx=None,
    uidx=None,
    grid=None,
    xscl=None,
    uscl=None,
):
    """
    Used by plot_trajectory to plot a single trajectory.
    """
    if xidx is None:
        dim_x = traj.shape[1]
        idx_x = np.arange(dim_x)
    else:
        idx_x = np.array(xidx)
        dim_x = len(idx_x)

    if us is None:
        dim_u = 0
    else:
        assert traj.shape[0] == us.shape[0], (
            "Trajectory and control input arrays must have the same time dimension"
        )
        if uidx is None:
            dim_u = us.shape[1]
            idx_u = np.arange(dim_u)
        else:
            idx_u = np.array(uidx)
            dim_u = len(idx_u)

    # Trim time array if needed
    if len(ts) > traj.shape[0]:
        ts = ts[: traj.shape[0]]

    # Set up subplot layout from metadata or use default
    if grid is None:
        # Default: one column with a row per state
        n_rows, n_cols = dim_x + dim_u, 1
        fig_size = (6, n_rows * 2)
    else:
        n_rows, n_cols = grid
        fig_size = (3 * n_cols, 2.5 * n_rows)

    if axes is None:
        # Create subplots
        fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)
        if n_rows * n_cols == 1:
            ax = [ax]  # Make it iterable for single subplot
        else:
            ax = ax.flatten()
    else:
        # Use provided axes
        fig = axes[0].figure
        ax = axes

    # Plot each state
    for i in range(dim_x):
        ax[i].plot(
            ts,
            traj[:, idx_x[i]],
            LINESTY[idx % 4],
            color=PALETTE[idx % 6],
            linewidth=2,
            label=label,
        )
        ax[i].set_xlim([0, ts[-1]])
        ax[i].grid(True, alpha=0.3)

        ax[i].set_ylabel(f"State {idx_x[i] + 1}", fontsize=10)
        if i == 0:  # Only show legend on first subplot
            ax[i].legend(loc="best", fontsize=9)

        if axes is None:
            _scale_axes(ax[i], traj[:, idx_x[i]], xscl)

    if dim_u > 0:
        us_arr = cast(np.ndarray, us)
        # Plot only once as this is from data
        offset = dim_x
        for i in range(dim_u):
            ax[offset + i].plot(ts, us_arr[:, idx_u[i]], "-", color="#3498db", linewidth=2)
            ax[offset + i].set_xlim([0, ts[-1]])
            ax[offset + i].grid(True, alpha=0.3)
            ax[offset + i].set_ylabel(f"Control {i + 1}", fontsize=10)

            _scale_axes(ax[offset + i], us_arr[:, idx_u[i]], uscl)

    for i in range(n_cols):
        ax[-i - 1].set_xlabel("Time", fontsize=10)
        ax[-i - 1].set_xlim([2 * ts[0] - ts[1], 2 * ts[-1] - ts[-2]])

    return fig, ax


def plot_summary(npz_files, labels=None, ifscl=True, ifclose=True, prefix=".", output_path=None):
    """
    Plot training losses and prediction criterion for multiple summary files on the same figure.

    Args:
        npz_files (list): List of NPZ file paths containing summary data.
        labels (list): List of labels for each run (optional).
        ifscl (bool): If True, scale the loss by the first epoch loss.
        ifclose (bool): Whether to close the plot after saving.
        prefix (str): Directory prefix used when output_path is not an absolute path.
        output_path (str, optional): If provided, save the figure to this path. By default the
            plot is not written to disk.
    """
    _files = [f"{npz}/{npz}_summary.npz" for npz in npz_files]
    npzs = [np.load(_f, allow_pickle=True) for _f in _files]
    ax = None
    for idx, npz in enumerate(npzs):
        _, ax = plot_one_summary(npz, label=str(idx), index=idx, ifscl=ifscl, axes=ax)

    lbls = labels if labels is not None else [f"Run {i + 1}" for i in range(len(npzs))]
    titl = ", ".join([f"{i}: {l}" for i, l in enumerate(lbls)])
    assert ax is not None
    ax[0].set_title(ax[0].get_title() + "\n" + titl)

    plt.tight_layout()
    if output_path is not None:
        save_path = output_path
        if not os.path.isabs(save_path):
            save_path = os.path.join(prefix, save_path)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    if ifclose:
        plt.close()

    return npzs


def plot_one_summary(npz, label="", index=0, ifscl=True, axes=None):
    """
    Plot training losses and prediction criterion from a summary file.

    Args:
        npz (dict): Dictionary from the NPZ file.
        label (str): Label for the plot legend.
        index (int): Index to select color from the PALETTE.
        ifscl (bool): If True, scale the loss by the first epoch loss.
        axes (list, optional): List of axes to plot on. If None, creates new subplots.
    """
    clr = PALETTE[index % len(PALETTE)]

    if axes is None:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
    else:
        fig = axes[0].figure
        ax = axes
    scl = npz["hist"][0]["train_total"][0] if ifscl else 1.0
    for hist in npz["hist"]:
        key = ["total", "dynamics"]
        for _k in hist.keys():
            if "train" in _k:
                if _k[6:] not in key:
                    key.append(_k[6:])
        epo = hist["epoch"]
        for _i, _k in enumerate(key):
            _sty = LINESTY[_i % len(LINESTY)]
            ax[0].semilogy(
                epo,
                np.abs(hist[f"train_{_k}"]) / scl,
                _sty,
                color=clr,
                label=f"{label} T {_k[:3]}",
                linewidth=1.5,
            )
            ax[0].semilogy(
                epo,
                np.abs(hist[f"valid_{_k}"]) / scl,
                _sty,
                color=clr,
                label=f"{label} V {_k[:3]}",
                linewidth=0.75,
            )
    if ifscl:
        ax[0].set_title("Training Loss (scaled)")
        ax[0].set_ylabel("Relative Loss")
    else:
        ax[0].set_title("Training Loss (raw)")
        ax[0].set_ylabel("Loss")
    ax[0].legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5))

    e_crit, h_crit, n_crit = npz["crit_epoch"], npz["crits"], npz["crit_name"]
    if len(e_crit) > 0:
        ax[1].semilogy(e_crit, np.abs(h_crit[0]), "-", color=clr, label=f"{label} Train")
        ax[1].semilogy(e_crit, np.abs(h_crit[1]), "--", color=clr, label=f"{label} Valid")
        ax[1].legend()
    ax[1].set_title("Prediction Criterion")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(f"Criterion {n_crit}")

    return fig, ax


def plot_hist(hist, crit, crit_name, model_name, ifclose=True, prefix="."):
    tmp = np.array(crit).T
    npz = {
        "hist": hist,
        "crit_epoch": tmp[0] if len(tmp) > 0 else [],
        "crits": tmp[1:] if len(tmp) > 1 else [],
        "crit_name": crit_name,
    }
    _ = plot_one_summary(npz, label="", index=0, ifscl=False, axes=None)

    plt.tight_layout()
    plt.savefig(
        f"{prefix}/{model_name}_history.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    if ifclose:
        plt.close()


def _resolve_cv_results_path(cv_file):
    if cv_file.endswith(".npz"):
        return cv_file

    cv_dir = os.path.dirname(cv_file)
    cv_name = os.path.basename(cv_file)
    candidates = [
        os.path.join(cv_dir, f"{cv_name}_cv.npz"),
        os.path.join(cv_file, f"{cv_name}_cv.npz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def plot_search_results(
    params,
    means,
    *,
    key_labels,
    metric_name,
    best_idx,
    stds=None,
    mode=None,
    title="Hyperparameter Search",
    output_path=None,
    ifclose=True,
    value_scale="log",
    axis_scales=None,
):
    """
    Plot hyperparameter-search results from already-collected arrays.
    """
    params = np.asarray(params)
    means = np.asarray(means, dtype=float)
    if stds is None:
        stds = np.zeros_like(means)
    else:
        stds = np.asarray(stds, dtype=float)
    axis_scales = list(axis_scales) if axis_scales is not None else []
    if mode is None:
        if params.ndim == 1:
            mode = "1d"
        elif params.ndim == 2 and params.shape[1] == 2:
            mode = "2d"
        elif params.ndim == 2 and params.shape[1] == 3:
            mode = "3d"
        else:
            mode = "history"
    best_label = f"Best: {metric_name} = {means[best_idx]:.3e} ± {stds[best_idx]:.3e}"

    if mode == "1d":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(params, means, "ko", markerfacecolor="none", label="Mean", markersize=8)
        ax.errorbar(params, means, yerr=stds, fmt="o", color="black", capsize=5, label="Std Dev")
        ax.plot(params[best_idx], means[best_idx], "rs", label=best_label, markersize=8)
        if value_scale == "log" and np.all(means > 0.0):
            ax.set_yscale(value_scale)
        _apply_search_axis_scale(ax, "x", params, axis_scales, 0)
        ax.set_xticks(params)
        ax.set_xlabel(key_labels[0])
        ax.set_ylabel(metric_name)
        ax.legend()

    elif mode == "2d":
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(params[:, 0], params[:, 1], c=means, s=100, cmap="viridis")
        ax.scatter(
            params[best_idx, 0],
            params[best_idx, 1],
            facecolors="none",
            edgecolors="red",
            s=150,
            marker="s",
            linewidths=2,
            label=best_label,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sc, cax=cax)
        if value_scale == "log" and np.all(means > 0.0) and means.max() > means.min():
            sc.set_norm(colors.LogNorm(vmin=means.min(), vmax=means.max()))
        cbar.set_label(f"Mean {metric_name}")
        _apply_search_axis_scale(ax, "x", params[:, 0], axis_scales, 0)
        _apply_search_axis_scale(ax, "y", params[:, 1], axis_scales, 1)
        ax.set_xticks(np.unique(params[:, 0]))
        ax.set_yticks(np.unique(params[:, 1]))
        ax.set_xlabel(key_labels[0])
        ax.set_ylabel(key_labels[1])
        ax.legend()

    elif mode == "3d":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax3d = cast(Any, ax)
        p = ax3d.scatter(
            params[:, 0], params[:, 1], zs=params[:, 2], c=means, s=100, cmap="viridis"
        )
        ax3d.scatter(
            params[best_idx, 0],
            params[best_idx, 1],
            zs=params[best_idx, 2],
            c="red",
            s=150,
            marker="s",
            label=best_label,
        )
        cbar = fig.colorbar(p)
        cbar.set_label(f"Mean {metric_name}")
        ax3d.set_xticks(np.unique(params[:, 0]))
        ax3d.set_yticks(np.unique(params[:, 1]))
        ax3d.set_zticks(np.unique(params[:, 2]))
        _apply_search_axis_scale(ax3d, "x", params[:, 0], axis_scales, 0)
        _apply_search_axis_scale(ax3d, "y", params[:, 1], axis_scales, 1)
        _apply_search_axis_scale(ax3d, "z", params[:, 2], axis_scales, 2)
        ax3d.set_xlabel(key_labels[0])
        ax3d.set_ylabel(key_labels[1])
        ax3d.set_zlabel(key_labels[2])
        ax3d.legend()

    elif mode == "history":
        fig, ax = plt.subplots(figsize=(8, 6))
        indices = np.arange(len(means))
        ax.plot(indices, means, "ko-", markerfacecolor="none", label="Mean", markersize=6)
        ax.plot(indices[best_idx], means[best_idx], "rs", label=best_label, markersize=8)
        if value_scale == "log" and np.all(means > 0.0):
            ax.set_yscale(value_scale)
        ax.set_xlabel("Evaluation Index")
        ax.set_ylabel(metric_name)
        ax.legend()
    else:
        raise ValueError(f"Unsupported search-results plot mode: {mode}")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        output_path = os.fspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    if ifclose:
        plt.close()

    return fig, ax


def _apply_search_axis_scale(ax, axis, values, axis_scales, index):
    if index >= len(axis_scales) or axis_scales[index] != "log":
        return
    values = np.asarray(values, dtype=float)
    if not np.all(values > 0.0):
        return
    if axis == "x":
        ax.set_xscale("log")
    elif axis == "y":
        ax.set_yscale("log")
    elif axis == "z":
        ax.set_zscale("log")


def plot_cv_results(cv_file, keys=None, ifclose=True, prefix=".", value_scale="log"):
    """
    Plot cross-validation results.

    Args:
        cv_results (list): List of CVResult objects.
        metric_name (str): Name of the metric to plot.
        ifclose (bool): Whether to close the plot after saving.
        prefix (str): Directory prefix for saving the plot.
    """
    if keys is None:
        params, metrics, metric_name, best_idx = _collect_cv_results(cv_file, [])
        params = np.arange(len(metrics))
        mode = "1d"
        key_labels = ["Hyperparameter Index"]
    elif len(keys) < 4:
        params, metrics, metric_name, best_idx = _collect_cv_results(cv_file, keys)
        if len(keys) == 1:
            params = params.flatten()
            mode = "1d"
        elif len(keys) == 2:
            mode = "2d"
        elif len(keys) == 3:
            mode = "3d"
        key_labels = [_k.split(".")[-1] for _k in keys]
    else:
        raise ValueError("Can only plot up to 3 hyperparameters at once.  Try keys=None.")
    means = metrics[:, 0]
    stds = metrics[:, 1]
    return plot_search_results(
        params,
        means,
        key_labels=key_labels,
        metric_name=metric_name,
        best_idx=best_idx,
        stds=stds,
        mode=mode,
        title=f"Cross-Validation Results - Metric: {metric_name}",
        output_path=f"{prefix}/cv_results.png",
        ifclose=ifclose,
        value_scale=value_scale,
    )


def _collect_cv_results(cv_file, keys):
    tmp = np.load(_resolve_cv_results_path(cv_file), allow_pickle=True)
    cv_res = tmp["all_results"]
    metric_name = tmp["metric_name"]
    best_idx = tmp["best_idx"]

    _keys = []
    for _k in keys:
        if _k.startswith("training"):
            # Legacy key update
            _ss = _k.split(".")
            _ss[0] = "phases.0"
            _keys.append(".".join(_ss))
        else:
            _keys.append(_k)

    params, metrics = [], []
    for res in cv_res:
        params.append([res.params[k] for k in _keys])
        metrics.append([res.mean_metric, res.std_metric])

    return np.array(params), np.array(metrics), metric_name, best_idx


def _get_contour_func(ax, mode):
    if mode == "contour":
        contour_func = ax.contour
    elif mode == "contourf":
        contour_func = ax.contourf
    elif mode == "tricontourf":
        contour_func = ax.tricontourf
    return contour_func


def plot_contour(
    arrays,
    x=None,
    y=None,
    vmin=None,
    vmax=None,
    levels=20,
    figsize=(12, 4),
    colorbar=True,
    axes=None,
    label=None,
    grid=None,
    mode="contourf",
    **kwargs,
):
    """
    Plot a grid of contour plots for a list of 2D arrays.

    Parameters:
        arrays (list of np.ndarray): List of 2D arrays to plot.
        titles (list of str): Titles for each subplot.
        x, y (np.ndarray): Optional meshgrid for axes.
        vmin, vmax (float): Color limits.
        levels (int or array): Contour levels.
        figsize (tuple): Figure size.
        colorbar (bool): Whether to add a colorbar.
        axes: Existing figure/axes tuple.
        label (list): Titles for each subplot.
        grid (tuple): Grid layout (n_rows, n_cols).
        mode (str): 'contour', 'contourf', or 'tricontourf'.
        **kwargs: Additional arguments for contourf.

    Returns:
        fig, axes: Matplotlib figure and axes.
    """
    # Validate inputs
    n = len(arrays)
    if grid is None:
        grid = (1, n)
    assert grid[0] * grid[1] >= n, "Grid size too small for number of arrays"

    if label is not None:
        assert len(label) == n, "Number of labels must match number of arrays, if provided"

    if axes is None:
        fig, ax = plt.subplots(grid[0], grid[1], figsize=figsize, sharex=True, sharey=True)
    else:
        fig, ax = axes

    assert mode in ["contour", "contourf", "tricontourf"], (
        "Mode must be 'contour', 'contourf', or 'tricontourf'"
    )
    if mode == "tricontourf":
        assert x is not None and y is not None, "x and y must be provided for tricontourf"

    # Prepare contour arguments
    contour_args = {}
    contour_args.update(**kwargs)
    if isinstance(levels, int):
        vmin = arrays.min() if vmin is None else vmin
        vmax = arrays.max() if vmax is None else vmax
        _lvls = np.linspace(vmin, vmax, levels)
    elif isinstance(levels, (list, np.ndarray)):
        _lvls = levels
    else:
        raise ValueError("Levels must be an integer or a list/array of levels")
    contour_args = {"levels": _lvls}

    # Plotting
    ims = []
    _ax = ax.flatten() if grid[0] * grid[1] > 1 else [ax]
    for i, arr in enumerate(arrays):
        _func = _get_contour_func(_ax[i], mode)
        if x is not None and y is not None:
            im = _func(x, y, arr, **contour_args)
        else:
            im = _func(arr, **contour_args)
        ims.append(im)
        if label:
            _ax[i].set_title(label[i])
    if colorbar:
        fig.colorbar(ims[0], ax=ax, orientation="vertical", fraction=0.02, pad=0.04)

    return fig, ax


def compare_contour(
    x_true,
    x_pred,
    x=None,
    y=None,
    vmin=None,
    vmax=None,
    levels=20,
    figsize=(12, 4),
    colorbar=True,
    axes=None,
    label=None,
    mode="contourf",
    **kwargs,
):
    """
    Compare two contours with error contours.
    """
    vmin = x_true.min() if vmin is None else vmin
    vmax = x_true.max() if vmax is None else vmax
    x_diff = x_true - x_pred
    err = np.linalg.norm(x_diff) / np.linalg.norm(x_true)
    label = ["Truth", "Reconstructed", f"Error: {err * 100:4.2f}%"]
    arrays = [x_true, x_pred, x_diff]
    return plot_contour(
        arrays,
        x=x,
        y=y,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        figsize=figsize,
        colorbar=colorbar,
        axes=axes,
        label=label,
        grid=(1, 3),
        mode=mode,
        **kwargs,
    )


def animate(fig_func, filename, fps=10, n_frames=None, writer_args=None, fig_args=None):
    """
    Create an animation by calling a figure-generating function for each frame.

    Args:
        fig_func (function): Function that generates a figure for a given frame index.
                             It should accept the frame index as its first argument,
                             and return a matplotlib figure and axes.
        filename (str): Output filename for the output file.
        fps (int): Frames per second for the animation.
        n_frames (int): Total number of frames in the animation.
        writer_args (dict): Additional keyword arguments to pass to imageio.get_writer.
        fig_args (dict): Additional keyword arguments to pass to fig_func.
    """
    if fig_args is None:
        fig_args = {}
    if writer_args is None:
        writer_args = {}
    writer = imageio.get_writer(filename, fps=fps, **writer_args)
    if n_frames is None:
        raise ValueError("n_frames must be provided.")
    for j in range(n_frames):
        logger.info(f"Generating frame {j + 1}/{n_frames} for {filename}")

        fig, ax = fig_func(j, **fig_args)

        # Render the figure to a numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(2 * h, 2 * w, 4)
        writer.append_data(frame[:, :, :3])

        # Removing existing objects otherwise they accumulate in canvas.draw
        for _a in ax.flatten():
            for _c in _a.collections:
                _c.remove()
        plt.close(fig)
    writer.close()
    logger.info(f"Animation saved to {filename}")
