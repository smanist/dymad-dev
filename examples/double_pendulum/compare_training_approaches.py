"""
Compare NODE vs Weak Form training approaches using the unified LDM model.

This script trains both methods sequentially and creates final comparison plots.
"""

from pathlib import Path
import sys, logging
import yaml
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_comparison.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from src.training.node_trainer import NODETrainer
from src.training.weak_form_trainer import WeakFormTrainer
from src.utils.plot import plot_hist

def load_and_modify_config(method_name):
    """Load base config and modify for specific training method."""
    config_path = 'config_training_compare.yaml'
    
    # Load base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for specific method
    config['weak_form']['enabled'] = (method_name == 'weak_form')
    config['model']['name'] = f'comparison_{method_name}'
    
    return config

def train_single_method(trainer_class, method_name, method_key):
    """Train a single method with default plotting."""
    logger.info(f"{'='*50}")
    logger.info(f"Starting {method_name} Training")
    logger.info(f"{'='*50}")
    
    # Load and modify config for this method
    config = load_and_modify_config(method_key)
    
    # Create temporary config file for trainer
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    config_path = temp_file.name
    
    try:
        # Create trainer
        trainer = trainer_class(config_path)
        
        logger.info(f"{method_name} - Model parameters: {sum(p.numel() for p in trainer.model.parameters())}")
        logger.info(f"{method_name} - Training samples: {len(trainer.train_set)}")
        logger.info(f"{method_name} - Validation samples: {len(trainer.validation_set)}")
        logger.info(f"{method_name} - Test samples: {len(trainer.test_set)}")
        
        # Training parameters
        n_epochs = trainer.config['training']['n_epochs']
        save_interval = trainer.config['training']['save_interval']  
        
        logger.info(f"{method_name} - Starting training for {n_epochs} epochs with plots every {save_interval} epochs")
        overall_start_time = time.time()
        
        # Training metrics tracking
        epoch_times = []
        best_val_loss = float('inf')
        convergence_epoch = None
        
        for epoch in range(trainer.start_epoch, n_epochs):
            epoch_start_time = time.time()
            
            # Training and evaluation
            train_loss = trainer.train_epoch()
            val_loss = trainer.evaluate(trainer.validation_loader)
            test_loss = trainer.evaluate(trainer.test_loader)
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Record history
            trainer.hist.append([train_loss, val_loss, test_loss])
            
            # Plot training history every epoch using plot.py function
            plot_hist(trainer.hist, epoch + 1, f"{trainer.model_name}")
            
            # Check for convergence (improvement in validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                convergence_epoch = epoch + 1
            
            # Periodic logging and plotting
            if (epoch + 1) % save_interval == 0:
                test_rmse = trainer.evaluate_rmse('test', plot=True)  # Use default plotter
                avg_epoch_time = np.mean(epoch_times[-save_interval:])
                
                logger.info(
                    f"{method_name} - Epoch {epoch+1:4d}/{n_epochs} | "
                    f"Train: {train_loss:.4e} | "
                    f"Val: {val_loss:.4e} | "
                    f"Test: {test_loss:.4e} | "
                    f"RMSE: {test_rmse:.4e} | "
                    f"Epoch Time: {avg_epoch_time:.2f}s"
                )
            
            # Save best model
            trainer.save_if_best(val_loss, epoch)
            
            # Periodic checkpoint
            if (epoch + 1) % trainer.config['training']['save_interval'] == 0:
                trainer.save_checkpoint(epoch)
        
        total_training_time = time.time() - overall_start_time
        avg_epoch_time = np.mean(epoch_times)
        
        # Final evaluation with plots
        final_train_loss = trainer.evaluate(trainer.train_loader)
        final_val_loss = trainer.evaluate(trainer.validation_loader)
        final_test_loss = trainer.evaluate(trainer.test_loader)
        final_test_rmse = trainer.evaluate_rmse('test', plot=True)  # Final prediction plot
        
        results = {
            'method': method_name,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_test_loss': final_test_loss,
            'final_test_rmse': final_test_rmse,
            'best_val_loss': best_val_loss,
            'convergence_epoch': convergence_epoch,
            'trainer': trainer,
            'config': config
        }
        
        logger.info(f"\n{method_name} - Final Results:")
        logger.info(f"  Total training time: {total_training_time:.2f} seconds")
        logger.info(f"  Average epoch time: {avg_epoch_time:.2f} seconds")
        logger.info(f"  Final train loss: {final_train_loss:.4e}")
        logger.info(f"  Final validation loss: {final_val_loss:.4e}")
        logger.info(f"  Final test loss: {final_test_loss:.4e}")
        logger.info(f"  Final test RMSE: {final_test_rmse:.4e}")
        logger.info(f"  Best validation loss: {best_val_loss:.4e} (epoch {convergence_epoch})")
        
        return results
        
    finally:
        # Clean up temporary config file
        if os.path.exists(config_path):
            os.unlink(config_path)

def create_final_comparison_plots(node_results, weak_results):
    """Create final comparison plots for both methods."""
    logger.info("Creating final comparison plots...")
    
    node_trainer = node_results['trainer']
    weak_trainer = weak_results['trainer']
    
    # 1. Training History Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training History Comparison: NODE vs Weak Form', fontsize=14)
    
    node_hist = np.array(node_trainer.hist)
    weak_hist = np.array(weak_trainer.hist)
    epochs = np.arange(1, len(node_hist) + 1)
    
    # Training Loss
    axes[0, 0].semilogy(epochs, node_hist[:, 0], 'b-', label='NODE', linewidth=2)
    axes[0, 0].semilogy(epochs, weak_hist[:, 0], 'r-', label='Weak Form', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0, 1].semilogy(epochs, node_hist[:, 1], 'b-', label='NODE', linewidth=2)
    axes[0, 1].semilogy(epochs, weak_hist[:, 1], 'r-', label='Weak Form', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test Loss
    axes[1, 0].semilogy(epochs, node_hist[:, 2], 'b-', label='NODE', linewidth=2)
    axes[1, 0].semilogy(epochs, weak_hist[:, 2], 'r-', label='Weak Form', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test Loss')
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary metrics
    axes[1, 1].axis('off')
    summary_text = f"""
    Final Results Summary:
    
    NODE:
    • Training Time: {node_results['total_training_time']:.2f}s
    • Final Test Loss: {node_results['final_test_loss']:.4e}
    • Final Test RMSE: {node_results['final_test_rmse']:.4e}
    
    Weak Form:
    • Training Time: {weak_results['total_training_time']:.2f}s
    • Final Test Loss: {weak_results['final_test_loss']:.4e}
    • Final Test RMSE: {weak_results['final_test_rmse']:.4e}
    
    Speed Ratio: {weak_results['total_training_time']/node_results['total_training_time']:.2f}x
    """
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('final_training_history_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction Comparison
    # Get a test trajectory (both should have same data)
    test_trajectory = node_trainer.test_set[0]
    
    # Extract states and controls
    states = test_trajectory[:, :node_trainer.metadata['n_state_features']]
    controls = test_trajectory[:, -node_trainer.metadata['n_control_features']:]
    
    # Initial condition
    x0 = states[0]
    ts = node_trainer.t
    
    # Get predictions from both models
    with torch.no_grad():
        node_pred = node_trainer.model.predict(x0, controls, ts)
        weak_pred = weak_trainer.model.predict(x0, controls, ts)
    
    # Convert to numpy for plotting
    states_np = states.cpu().numpy()
    node_pred_np = node_pred.cpu().numpy()
    weak_pred_np = weak_pred.cpu().numpy()
    ts_np = ts.cpu().numpy()
    
    # Create prediction comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Final Prediction Comparison: NODE vs Weak Form', fontsize=14)
    
    state_names = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
    
    for i in range(4):
        ax = axes[i//2, i%2]
        ax.plot(ts_np, states_np[:, i], 'k-', label='Ground Truth', linewidth=2)
        ax.plot(ts_np, node_pred_np[:, i], 'b--', label='NODE', linewidth=1.5)
        ax.plot(ts_np, weak_pred_np[:, i], 'r:', label='Weak Form', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(state_names[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and log final errors
    node_error = np.mean((states_np - node_pred_np)**2)
    weak_error = np.mean((states_np - weak_pred_np)**2)
    
    logger.info(f"Final prediction MSE comparison:")
    logger.info(f"  NODE: {node_error:.4e}")
    logger.info(f"  Weak Form: {weak_error:.4e}")
    logger.info(f"  Ratio (Weak/NODE): {weak_error/node_error:.4e}")

def print_summary(node_results, weak_results):
    """Print a comprehensive summary comparison."""
    logger.info(f"{'='*60}")
    logger.info("TRAINING COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    
    logger.info(f"{'Metric':<25} {'NODE':<15} {'Weak Form':<15} {'Ratio (W/N)':<12}")
    logger.info(f"{'-'*67}")
    
    metrics = [
        ('Total Training Time (s)', 'total_training_time'),
        ('Avg Epoch Time (s)', 'avg_epoch_time'),
        ('Final Train Loss', 'final_train_loss'),
        ('Final Val Loss', 'final_val_loss'),
        ('Final Test Loss', 'final_test_loss'),
        ('Final Test RMSE', 'final_test_rmse'),
        ('Best Val Loss', 'best_val_loss'),
        ('Convergence Epoch', 'convergence_epoch')
    ]
    
    for name, key in metrics:
        node_val = node_results[key]
        weak_val = weak_results[key]
        
        if key == 'convergence_epoch':
            logger.info(f"{name:<25} {node_val:<15} {weak_val:<15} {'-':<12}")
        else:
            ratio = weak_val / node_val if node_val != 0 else float('inf')
            logger.info(f"{name:<25} {node_val:<15.4e} {weak_val:<15.4e} {ratio:<12.4e}")
    
    logger.info(f"{'-'*67}")
    logger.info("\nKey Insights:")
    
    time_ratio = weak_results['total_training_time'] / node_results['total_training_time']
    if time_ratio < 1:
        logger.info(f"• Weak form is {1/time_ratio:.2f}x faster to train overall")
    else:
        logger.info(f"• Weak form is {time_ratio:.2f}x slower to train overall")
    
    logger.info(f"• NODE achieves {node_results['final_test_rmse']:.4e} final test RMSE")
    logger.info(f"• Weak form achieves {weak_results['final_test_rmse']:.4e} final test RMSE")
    
    if weak_results['final_test_rmse'] < node_results['final_test_rmse']:
        logger.info("• Weak form has better final test accuracy!")
    else:
        logger.info("• NODE has better final test accuracy")

def main():
    """Main comparison function with sequential training."""
    logger.info("Unified LDM Model: Sequential Training Method Comparison")
    logger.info("Comparing NODE (direct ODE) vs Weak Form training")
    
    # Train both methods sequentially
    logger.info("Training NODE method first...")
    node_results = train_single_method(NODETrainer, "NODE (Direct ODE)", "node")
    
    logger.info("Training Weak Form method second...")
    weak_results = train_single_method(WeakFormTrainer, "Weak Form", "weak_form")
    
    # Create final comparison plots
    create_final_comparison_plots(node_results, weak_results)
    
    # Print comprehensive summary
    print_summary(node_results, weak_results)
    
    logger.info(f"{'='*60}")
    logger.info("Sequential training comparison complete!")
    logger.info("Final plots created:")
    logger.info("  - final_training_history_comparison.png: Training history comparison")
    logger.info("  - final_prediction_comparison.png: Prediction comparison")
    logger.info("  - Individual training history plots from default plotters")
    logger.info("  - Individual prediction plots from default plotters")
    logger.info("All training logs saved to 'training_comparison.log'")

if __name__ == "__main__":
    main() 