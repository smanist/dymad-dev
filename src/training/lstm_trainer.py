import torch
from typing import Tuple

from .trainer_base import TrainerBase
from src.models.lstm import LSTM

class LSTMTrainer(TrainerBase):
    """Trainer class for LSTM discrete-time models."""
    
    def __init__(self, config_path: str):
        """Initialize LSTM trainer with configuration."""
        super().__init__(config_path, LSTM)
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            # Extract states and controls
            states = batch[:, :, :self.metadata['n_state_features']]
            controls = batch[:, :, -self.metadata['n_control_features']:]
            
            # For LSTM, we want to predict the next state given the current states
            # Input: states[:, :-1], controls[:, :-1]
            # Target: states[:, 1:]
            inputs_states = states[:, :-1]
            inputs_controls = controls[:, :-1]
            target_states = states[:, 1:]
            
            # Forward pass
            _, _, predicted_states = self.model(inputs_states, inputs_controls)
            
            # Compute loss
            loss = self.criterion(predicted_states, target_states[:, -1])
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.scheduler.step()
        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        return total_loss / len(self.train_loader)
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the provided dataloader."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Extract states and controls
                states = batch[:, :, :self.metadata['n_state_features']]
                controls = batch[:, :, -self.metadata['n_control_features']:]
                
                # For LSTM, we want to predict the next state given the current states
                inputs_states = states[:, :-1]
                inputs_controls = controls[:, :-1]
                target_states = states[:, 1:]
                
                # Forward pass
                _, _, predicted_states = self.model(inputs_states, inputs_controls)
                
                # Compute loss
                loss = self.criterion(predicted_states, target_states[:, -1])
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def evaluate_rollout_rmse(self, split: str = 'test', plot: bool = False) -> float:
        """
        Calculate RMSE using multi-step rollout prediction on a random trajectory.
        This evaluates the discrete-time model's ability to roll out longer sequences.
        """
        dataset = getattr(self, f"{split}_set")
        trajectory = random.choice(dataset)
        
        # Extract states and controls
        states = trajectory[:, :self.metadata['n_state_features']]
        controls = trajectory[:, -self.metadata['n_control_features']:]
        
        # Get initial state and use it for rollout prediction
        initial_state = states[0].unsqueeze(0)  # Add batch dimension
        
        # Generate predictions using rollout
        predictions = self.model.predict(initial_state, controls.unsqueeze(0), self.t)
        
        # Remove batch dimension for comparison
        predictions = predictions[:, 0, :]
        
        # Calculate RMSE
        from src.losses.evaluation import prediction_rmse
        rmse = prediction_rmse(
            self.model, trajectory, self.t, 
            self.metadata, f"{self.model_name}_{split}", plot=plot
        )
        
        return rmse
    
    def train(self) -> None:
        """Run full training loop with rollout RMSE evaluation."""
        from src.utils.plot import plot_hist
        
        n_epochs = self.config['training']['n_epochs']
        save_interval = self.config['training']['save_interval']
        
        for epoch in range(self.start_epoch, n_epochs):
            # Training and evaluation
            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.validation_loader)
            test_loss = self.evaluate(self.test_loader)
            
            # Record history
            self.hist.append([train_loss, val_loss, test_loss])
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{n_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}"
            )
            
            # Plotting
            plot_hist(self.hist, epoch+1, self.model_name)
            
            # Save best model
            self.save_if_best(val_loss, epoch)

            # Periodic checkpoint and evaluation
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)
                
                # Evaluate RMSE on random trajectories using rollout prediction
                rmse_train = self.evaluate_rollout_rmse('train', plot=False)
                rmse_val = self.evaluate_rollout_rmse('validation', plot=False)
                rmse_test = self.evaluate_rollout_rmse('test', plot=True)
            
                self.logger.info(
                    f"Rollout RMSE - "
                    f"Train: {rmse_train:.4f}, "
                    f"Validation: {rmse_val:.4f}, "
                    f"Test: {rmse_test:.4f}"
                )