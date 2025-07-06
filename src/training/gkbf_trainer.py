import torch, random
from typing import Type

from .trainer_base import TrainerBase
from ...src.data.trajectory_manager import TrajectoryManager
from ...src.losses.weak_form import weak_form_loss
from ...src.losses.evaluation import prediction_rmse_graph
from ...src.models.kbf import GKBF

class GKBFTrainer(TrainerBase):
    """Trainer class for graph Koopman bilinear form (GKBF) models."""

    def __init__(self, config_path: str, adj_mat: torch.Tensor, model_class: Type[torch.nn.Module] = GKBF):
        """Initialize wGKBF trainer with configuration.

        Args:
            config_path: Path to the configuration file
            adj_mat: Adjacency matrix tensor for the graph structure
        """
        self.adj_mat = adj_mat
        super().__init__(config_path, model_class)

    def _create_trajectory_manager(self):
        """Create TrajectoryManager with adjacency matrix for graph-based models."""
        return TrajectoryManager(self.metadata, device=self.device, adj=self.adj_mat)

    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for trajectory in self.train_loader:
            ## NOTE: KBF type models currently do not support batch training,
            # so we need to iterate over the trajectories (i.e. batch_size=1)
            batch = next(iter(trajectory)).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # Extract states and controls
            states = batch.x
            controls = batch.u
            edge_index = batch.edge_index
            # Forward pass
            predictions = self.model(states, controls, edge_index)
            # Use weak form loss
            loss = weak_form_loss(states, predictions, self.metadata['weak_dyn_param'], self.criterion)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        for scheduler in self.schedulers:
            scheduler.step()
        # Maintain minimum learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the provided dataloader."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for trajectory in dataloader:
                batch = next(iter(trajectory)).to(self.device)
                # Extract states and controls
                states = batch.x
                controls = batch.u
                edge_index = batch.edge_index
                # Forward pass
                predictions = self.model(states, controls, edge_index)
                # Use weak form loss
                loss = weak_form_loss(states, predictions, self.metadata['weak_dyn_param'], self.criterion)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def get_prediction_rmse_func(self):
        """Return the graph-specific prediction RMSE function."""
        from ...src.losses.evaluation import prediction_rmse_graph
        return prediction_rmse_graph
