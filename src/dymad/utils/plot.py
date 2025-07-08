import logging
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#000000", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
LINESTY = ["-", "--", "-.", ":"]

# Disable logging for matplotlib to avoid clutter in DEBUG mode
plt_logger = logging.getLogger('matplotlib')
plt_logger.setLevel(logging.INFO)

def plot_trajectory(traj, ts, model_name, metadata, us=None, labels=None, ifclose=True, prefix='.'):
    """
    Plot trajectories with optional control inputs and save the figure.

    Args:
        traj (np.ndarray): Trajectory data, shape (n_steps, n_features) or (n_traj, n_steps, n_features)
        ts (np.ndarray): Time points corresponding to the trajectory data, shape (n_steps,)
        model_name (str): Name of the model for the plot title and filename
        metadata (dict): Metadata containing configuration and state information
        us (np.ndarray, optional): Control inputs, shape (n_steps, n_controls)
        labels (list, optional): Labels for each trajectory, length must match number of trajectories
        ifclose (bool): Whether to close the plot after saving
        prefix (str): Directory prefix for saving the plot
    """
    if traj.ndim == 2:
        traj = np.array([traj])

    Ntrj = len(traj)
    assert Ntrj == len(labels), \
        "Number of trajectories must match number of labels"

    # Plot the first trajectory and create the axes
    fig, ax = plot_one_trajectory(traj[0], ts, metadata, idx=0, us=us, axes=None, label=labels[0])

    if Ntrj > 1:
        # Add additional trajectories to the same axes
        for i in range(1, Ntrj):
            rmse = np.linalg.norm(traj[0] - traj[i]) / (traj[0].shape[0] - 1)**0.5
            plot_one_trajectory(
                traj[i], ts, metadata, idx=i, us=None, axes=ax,
                label=labels[i]+f" rmse: {rmse:.4f}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{prefix}/{model_name}_prediction.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    if ifclose:
        plt.close()

def plot_one_trajectory(traj, ts, metadata, idx=0, us=None, axes=None, label=None):
    """
    Used by plot_trajectory to plot a single trajectory.
    """
    dim_x = traj.shape[1]

    if us is None:
        dim_u = 0
    else:
        assert traj.shape[0] == us.shape[0], \
            "Trajectory and control input arrays must have the same time dimension"
        dim_u = us.shape[1]

    # Trim time array if needed
    if len(ts) > traj.shape[0]:
        ts = ts[:traj.shape[0]]

    if axes is None:
        # Set up subplot layout from metadata or use default
        plotting_config = metadata.get('config', {}).get('plotting', {})
        if 'n_rows' in plotting_config and 'n_cols' in plotting_config:
            n_rows = plotting_config['n_rows']
            n_cols = plotting_config['n_cols']
            fig_size = (3 * n_cols, 2.5 * n_rows)
        else:
            # Default: one column with a row per state
            n_rows, n_cols = dim_x + dim_u, 1
            fig_size = (6, n_rows * 2)

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
        ax[i].plot(ts, traj[:, i], LINESTY[idx%4], color=PALETTE[idx%6], linewidth=2, label=label)
        ax[i].set_xlim([0, ts[-1]])
        ax[i].grid(True, alpha=0.3)

        ax[i].set_ylabel(f'State {i+1}', fontsize=10)
        if i == 0:  # Only show legend on first subplot
            ax[i].legend(loc='best', fontsize=9)

    if axes is None:
        for i in range(dim_x):
            # Set y-limits based on scaling mode (for normalized data)
            if 'scaler' in metadata:
                mode = metadata['scaler']['mode']
                if mode == "01":
                    ax[i].set_ylim([-0.1, 1.1])  # [0,1] range with small buffer
                elif mode == "-11":
                    ax[i].set_ylim([-1.2, 1.2])  # [-1,1] range with buffer
                elif mode == "std":
                    ax[i].set_ylim([-3, 3])      # ±3 std devs for standardized data
                else:  # mode=none
                    ymx = np.max(traj[:, i])
                    ymn = np.min(traj[:, i])
                    ax[i].set_ylim([ymn-0.1*abs(ymn), ymx+0.1*abs(ymx)])  # Use data range with buffer

    if axes is None and dim_u > 0:
        offset = dim_x
        for i in range(dim_u):
            ax[offset + i].plot(ts, us[:, i], '-', color='#3498db', linewidth=2)
            ax[offset + i].set_xlim([0, ts[-1]])
            ax[offset + i].grid(True, alpha=0.3)
            ax[offset + i].set_ylabel(f'Control {i+1}', fontsize=10)
    ax[-1].set_xlabel('Time', fontsize=10)

    return fig, ax

def plot_hist(hist, epoch, model_name, ifclose=True, prefix='.'):
    """
    Plot training history with loss curves for train/validation/test sets.

    Args:
        hist (list): History of losses, where hist[0] is epoch numbers and
                     hist[1:] contains losses for train, validation, and test sets.
        epoch (int): Number of epochs to plot.
        model_name (str): Name of the model for the plot title and filename.
        ifclose (bool): Whether to close the plot after saving.
        prefix (str): Directory prefix for saving the plot.
    """
    tmp = np.array(hist).T
    _e, _h = tmp[0][:epoch], tmp[1:,:epoch]

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot loss curves with modern styling
    plt.semilogy(_e, _h[0], '--', color='#3498db', linewidth=2,
                 label='Training', alpha=0.8)
    plt.semilogy(_e, _h[1], '-', color='#e74c3c', linewidth=2,
                 label='Validation', alpha=0.9)
    plt.semilogy(_e, _h[2], '-', color='#2ecc71', linewidth=2,
                 label='Test', alpha=0.9)

    # Styling
    plt.xlim([_e[0], _e[-1]+1])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    # Improve tick formatting
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Save with clean formatting
    plt.tight_layout()
    plt.savefig(f'{prefix}/{model_name}_history.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    if ifclose:
        plt.close()
