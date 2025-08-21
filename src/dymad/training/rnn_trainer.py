import logging
import time
import numpy as np
import random
import torch
from typing import Dict, List, Type, Union
import matplotlib.pyplot as plt

from dymad.data import DynData, DynGeoData
from dymad.training.trainer_base import TrainerBase
from dymad.utils.scheduler import make_scheduler
from dymad.utils import plot_hist

logger = logging.getLogger(__name__)

def _determine_chop_step(window: int, step: Union[int, float]) -> int:
    """
    Determine the chop step based on the window size and step value.
    """
    if isinstance(step, int):
        return step
    elif isinstance(step, float):
        stp = int(window * step)
        return min(max(stp, 1), window)
    else:
        raise ValueError(f"Invalid step type: {type(step)}. Expected int or float.")

class RNNTrainer(TrainerBase):
    """
    Trainer for the RNN model with LDM-like architecture and delay.
    
    This trainer handles the specific requirements of the RNN model:
    1. Processing sequences of states and controls with delay (L steps)
    2. Encoding L states to get L latent states
    3. Using an RNN with L layers to produce the next latent state (L+1)
    4. Decoding latent states z₂ to z_(L+1) to produce x_(L+1)
    5. Using the new u_(L+1) for the next iteration
    6. Calculating appropriate losses for sequential prediction
    """
    
    def __init__(self, config_path: str, model_class: Type[torch.nn.Module] = None, config_mod: Dict = None):
        """Initialize RNN trainer with configuration."""
        super().__init__(config_path, model_class, config_mod)

        # RNN-specific configuration parameters
        self.recon_weight = self.config['training'].get('reconstruction_weight', 1.0)
        self.dynamics_weight = self.config['training'].get('dynamics_weight', 1.0)
        self.multi_step_weight = self.config['training'].get('multi_step_weight', 1.0)  # Add this line

        self.time_delay = self.config['model'].get('time_delay', 1)
        self.multi_step = self.config['training'].get('multi_step', 3)

        # Configure sequence chopping for time-delayed inputs
        self.chop_mode = self.config['training'].get('chop_mode', 'initial')
        assert self.chop_mode in ['initial', 'unfold'], f"Invalid chop_mode: {self.chop_mode}"
        
        # Configure step size for sliding window
        self.chop_step = self.config['training'].get('chop_step', 1.0)
        assert self.chop_step > 0, f"Chop step must be positive. Got: {self.chop_step}"
                
        # Set up sweep scheduler if needed
        sweep_lengths = self.config['training'].get('sweep_lengths', [len(self.t)])
        epoch_step = self.config['training'].get('sweep_epoch_step', self.config['training']['n_epochs'])
        sweep_tols = self.config['training'].get('sweep_tols', None)
        sweep_mode = self.config['training'].get('sweep_mode', 'skip')
        
        # Add sweep scheduler if specified
        if sweep_lengths:
            self.schedulers.append(make_scheduler(
                scheduler_type="sweep", 
                sweep_lengths=sweep_lengths, 
                sweep_tols=sweep_tols,
                epoch_step=epoch_step, 
                mode=sweep_mode
            ))
        
        # Log additional information
        if self.chop_mode == 'initial':
            logger.info(f"Chop mode: {self.chop_mode}, initial steps only")
        else:
            logger.info(f"Chop mode: {self.chop_mode}, window stride: {self.chop_step}")
        
        logger.info(f"Loss weights - Dynamics: {self.dynamics_weight},"
                    f" Reconstruction: {self.recon_weight}")


            
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_dynamics_loss = 0.0
        total_encoder_recon_loss = 0.0
        total_decoder_recon_loss = 0.0
        min_lr = 5e-5
        
        batch_count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._process_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        avg_epoch_loss = total_loss / len(self.train_loader)

        for scheduler in self.schedulers:
            flag = scheduler.step(eploss=avg_epoch_loss)
            self.convergence_tolerance_reached = self.convergence_tolerance_reached or flag

        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        return avg_epoch_loss
    
    def train(self) -> None:
        """Run full training loop."""

        n_epochs = self.config['training']['n_epochs']
        save_interval = self.config['training']['save_interval']

        self.convergence_epoch = None
        self.epoch_times = []
        self.losshist = []

        overall_start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            # Training and evaluation
            # Only timing the train and validation phases
            # since test loss is only for reference
            epoch_start_time = time.time()

            train_loss= self.train_epoch()
            val_loss = self.evaluate(self.validation_loader)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            test_loss = self.evaluate(self.test_loader)

            # Record history
            self.hist.append([epoch, train_loss, val_loss, test_loss])
            self.losshist.append([
                epoch, 
                self.recon_loss.detach().cpu().item(), 
                self.dynamics_loss.detach().cpu().item(),
                1
            ])

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.start_epoch + n_epochs}, "
                f"Train Loss: {train_loss:.4e}, "
                f"Validation Loss: {val_loss:.4e}, "
                f"Test Loss: {test_loss:.4e}"
            )

            # Save best model
            self.save_if_best(val_loss, epoch)

            # Periodic checkpoint and evaluation
            if (epoch + 1) % save_interval == 0 or self.convergence_tolerance_reached:
                self.save_checkpoint(epoch)

                # Plot loss curves
                plot_hist(self.hist, epoch+1, self.model_name, prefix=self.results_prefix)
                plot_hist(self.losshist, epoch+1, "loss", prefix=self.results_prefix)

                # Evaluate RMSE on random trajectories
                train_rmse = self.evaluate_rmse('train', plot=False)
                val_rmse   = self.evaluate_rmse('validation', plot=False)
                test_rmse  = self.evaluate_rmse('test', plot=True)
                self.rmse.append([epoch, train_rmse, val_rmse, test_rmse])

                logger.info(
                    f"Prediction RMSE - "
                    f"Train: {train_rmse:.4e}, "
                    f"Validation: {val_rmse:.4e}, "
                    f"Test: {test_rmse:.4e}"
                )

                if self.convergence_tolerance_reached:
                    logger.info(f"Convergence reached at epoch {epoch+1} "
                                f"with validation loss {val_loss:.4e}")
                    break

        if self.rmse == []:
            train_rmse = self.evaluate_rmse('train', plot=False)
            val_rmse   = self.evaluate_rmse('validation', plot=False)
            test_rmse  = self.evaluate_rmse('test', plot=False)
            self.rmse.append([epoch, train_rmse, val_rmse, test_rmse])

        plot_hist(self.hist, epoch+1, self.model_name, prefix=self.results_prefix)
        total_training_time = time.time() - overall_start_time
        avg_epoch_time = np.mean(self.epoch_times)
        final_train_loss = self.evaluate(self.train_loader)
        final_val_loss = self.evaluate(self.validation_loader)
        final_test_loss = self.evaluate(self.test_loader)
        _ = self.evaluate_rmse('test', plot=True)

        # Process histories of loss and RMSE
        # These are saved in the checkpoint too, but here we process them for easier post-processing
        tmp = np.array(self.hist).T
        epoch_loss, losses = tmp[0], tmp[1:]
        tmp = np.array(self.rmse).T
        epoch_rmse, rmses = tmp[0], tmp[1:]

        # Save summary of training
        # Here we also save the model itself - a lazy approach but more "out-of-the-box" for deployment
        results = {
            'model_name': self.model_name,
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_test_loss': final_test_loss,
            'best_val_loss': self.best_loss,
            'convergence_epoch': self.convergence_epoch,
            'epoch_loss': epoch_loss,
            'losses': losses,
            'epoch_rmse': epoch_rmse,
            'rmses': rmses,
        }

        file_name = f'{self.results_prefix}/{self.model_name}_summary.npz'
        np.savez_compressed(file_name, **results)
        logger.info(f"Training complete. Summary of training:")
        for key in ['model_name', 'total_training_time', 'avg_epoch_time',
                    'final_train_loss', 'final_val_loss', 'final_test_loss',
                    'best_val_loss', 'convergence_epoch']:
            info = f"{results[key]:.4e}" if isinstance(results[key], float) else str(results[key])
            logger.info(f"{key}: {info}")
        logger.info(f"Summary and loss/rmse histories saved to {file_name}")

    def _process_batch(self, batch: Union[DynData, DynGeoData]) -> torch.Tensor:
        """Process a batch and return predictions and ground truth states."""
        num_steps = self.schedulers[1].get_length()

        if self.chop_mode == 'initial':
            B = batch.truncate(num_steps)
        else:
            B = batch.unfold(num_steps, _determine_chop_step(num_steps, self.chop_step))

        B = B.to(self.device)

        ts=self.t[:num_steps].to(self.device)

        z_seq = []
        x_recon= []


        for t in range(num_steps):
            x_t = B.x[:, t, :]  # (batch_size, n_total_state_features)
            u_t = B.u[:, t, :] if B.u is not None else None  # (batch_size, n_total_control_features)
            w_t = DynData(x=x_t, u=u_t)
            z_t = self.model.encoder(w_t)
            z_seq.append(z_t.unsqueeze(1))
            x_recon.append(self.model.decoder(z_t, w_t).unsqueeze(1))

        z_seq = torch.cat(z_seq, dim=1)  
        x_recon = torch.cat(x_recon, dim=1)


        z_pred = []
        x_pred = []

        for i in range(num_steps - self.time_delay):
            output, h_n = self.model.dynamics(z_seq[:, i:i+self.time_delay, :],w_t)
            z_pred_i = output[:,-1,:]
            x_pred_i = self.model.decoder(z_pred_i, w_t)
            z_pred.append(z_pred_i.unsqueeze(1))
            x_pred.append(x_pred_i.unsqueeze(1))
        
        z_pred = torch.cat(z_pred, dim=1)  # Shape: [batch_size, num_steps - time_delay, latent_dimension]
        x_pred = torch.cat(x_pred, dim=1)  # Shape: [batch_size, num_steps - time_delay, state_features]

        self.recon_loss = self.criterion(x_recon, B.x)
        self.dynamics_loss = self.criterion(x_pred, B.x[:, self.time_delay:, :])

        total_loss = self.dynamics_weight * self.dynamics_loss + self.recon_weight * self.recon_loss

        return total_loss


