import logging
import matplotlib.pyplot as plt
import numpy as np

# Disable logging for matplotlib to avoid clutter in DEBUG mode
plt_logger = logging.getLogger('matplotlib')
plt_logger.setLevel(logging.INFO)


def plot_trajectory(traj, ts, model_name, metadata):
    assert len(traj) == 2, "Expected exactly 2 trajectories (prediction and truth)"

    # Calculate RMSE
    rmse = np.linalg.norm(traj[0] - traj[1]) / (traj.shape[1] - 1)**0.5

    # Trim time array if needed
    if len(ts) > traj.shape[1]:
        ts = ts[:traj.shape[1]]

    # Set up subplot layout from metadata or use default
    plotting_config = metadata.get('config', {}).get('plotting', {})
    if 'n_rows' in plotting_config and 'n_cols' in plotting_config:
        n_rows = plotting_config['n_rows']
        n_cols = plotting_config['n_cols']
        fig_size = (3 * n_cols, 2.5 * n_rows)
    else:
        # Default: one column with a row per state
        n_rows, n_cols = metadata['n_state_features'], 1
        fig_size = (6, n_rows * 2)

    # Create subplots
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex=True)
    if metadata['n_state_features'] == 1:
        ax = [ax]  # Make it iterable for single subplot
    else:
        ax = ax.flatten()

    # Plot each state
    for i in range(metadata['n_state_features']):
        ax[i].plot(ts, traj[0, :, i], '-', color='#e74c3c', linewidth=2, label='Prediction')
        ax[i].plot(ts, traj[1, :, i], '--', color='#2c3e50', linewidth=2, label='Truth')
        ax[i].set_xlim([0, ts[-1]])
        ax[i].grid(True, alpha=0.3)

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
                ymx = np.max(traj[1, :, i])
                ymn = np.min(traj[1, :, i])
                ax[i].set_ylim([ymn-0.1*abs(ymn), ymx+0.1*abs(ymx)])  # Use data range with buffer

        ax[i].set_ylabel(f'State {i+1}', fontsize=10)
        if i == 0:  # Only show legend on first subplot
            ax[i].legend(loc='best', fontsize=9)

    # Set title and xlabel
    fig.suptitle(f'Trajectory Prediction (RMSE: {rmse:.4f})',
                 fontsize=12, fontweight='bold')
    ax[-1].set_xlabel('Time', fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'./{model_name}_prediction.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def plot_hist(hist, epoch, model_name):
    """Plot training history with loss curves for train/validation/test sets."""
    hist = np.array(hist)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot loss curves with modern styling
    epochs = np.arange(1, epoch + 1)
    plt.semilogy(epochs, hist[:epoch, 0], '--', color='#3498db', linewidth=2,
                 label='Training', alpha=0.8)
    plt.semilogy(epochs, hist[:epoch, 1], '-', color='#e74c3c', linewidth=2,
                 label='Validation', alpha=0.9)
    plt.semilogy(epochs, hist[:epoch, 2], '-', color='#2ecc71', linewidth=2,
                 label='Test', alpha=0.9)

    # Styling
    plt.xlim([1, epoch+2])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    # Improve tick formatting
    plt.tick_params(axis='both', which='major', labelsize=10)

    # Save with clean formatting
    plt.tight_layout()
    plt.savefig(f'./{model_name}_history.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()