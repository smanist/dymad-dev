# Double Pendulum: Unified LDM Training Comparison

This directory demonstrates training the same **Latent Dynamics Model (LDM)** architecture using two different training approaches:

1. **NODE Training** - Direct ODE integration loss (more accurate, slower)
2. **Weak Form Training** - Weak form loss (faster, approximate)

## Files

### Training Scripts
- `train_ldm_node.py` - Train LDM with NODE approach (direct ODE integration)
- `train_ldm_weak_form.py` - Train LDM with weak form approach  
- `compare_training_approaches.py` - **Compare both methods side-by-side**

### Configuration Files
- `config_ldm_node.yaml` - NODE-specific configuration
- `config_ldm_weak_form.yaml` - Weak form-specific configuration
- `config_training_compare.yaml` - Configuration for comparison script

## Quick Start

### 1. Compare Both Training Methods 
```bash
cd examples/double_pendulum
python compare_training_approaches.py
```

This will:
- Train the same LDM model with both approaches for **1000 epochs**
- Monitor progress every **10 epochs** with live plot updates
- Compare training time, accuracy, and predictions
- Generate comprehensive comparison plots and data export

**Output Files:**
- `training_comparison.log` - Complete training logs
- `training_comparison.png` - Prediction comparison plot
- `training_history_comparison.png` - Loss evolution and timing plots
- `training_history_comparison.csv` - Complete training metrics data

### 2. Train Individual Models

**NODE Training:**
```bash
python train_ldm_node.py
```
Uses `config_ldm_node.yaml`

**Weak Form Training:**
```bash
python train_ldm_weak_form.py
```
Uses `config_ldm_weak_form.yaml`

## Architecture

Both training scripts use the **same unified LDM model** but with different trainers:

- **NODETrainer**: Inherits from `TrainerBase`, uses direct ODE integration loss
- **WeakFormTrainer**: Inherits from `TrainerBase`, uses weak form loss
- **LDM Model**: Unified architecture with encoder, dynamics, and decoder networks

The `TrainerBase` class handles:
- Configuration loading
- Data setup and trajectory management
- Model initialization
- Training loop and checkpointing
- Evaluation and logging

## Comparison Features

The comparison script provides:

### Real-time Monitoring
- **Live plot updates** every 10 epochs during training
- **Normalized loss comparison** for fair evaluation
- **Per-epoch timing** measurements
- **Convergence analysis** with threshold tracking

### Comprehensive Analysis
- **Training efficiency**: Total time, per-epoch time, convergence speed
- **Final accuracy**: Test loss, RMSE, prediction quality
- **Convergence patterns**: When each method reaches accuracy thresholds
- **Visual comparison**: Side-by-side prediction plots

### Data Export
- **CSV export** of all training metrics
- **High-resolution plots** for publication
- **Detailed logging** with scientific notation formatting

## Key Insights

The unified LDM model allows direct comparison of training methodologies:

### NODE Training (Direct ODE Integration)
- **Pros**: More mathematically rigorous, potentially higher accuracy
- **Cons**: Slower training due to ODE solver calls
- **Use when**: Accuracy is critical, computational time is not a constraint
- **Config**: Includes ODE solver parameters (`ode_method`, `rtol`, `atol`)

### Weak Form Training  
- **Pros**: Faster training, good approximation of dynamics
- **Cons**: Approximate method, may sacrifice some accuracy
- **Use when**: Fast training is needed, approximate dynamics are acceptable
- **Config**: Includes weak form parameters (`reconstruction_weight`, `dynamics_weight`)

## Model Architecture

Both approaches use the identical LDM architecture:
- **Encoder**: Maps (state, control) → latent space
- **Dynamics**: Learns latent space derivatives  
- **Decoder**: Maps latent space → state space

The only difference is the training loss function:
- NODE: Integrates the learned dynamics and compares to ground truth
- Weak Form: Uses weak form PDE loss without explicit integration
