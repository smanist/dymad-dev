import torch, random

from .trainer_base import TrainerBase
from src.losses.weak_form import weak_form_loss
from src.models.wKBF import weakGraphKBF 

class wGKBFTrainer(TrainerBase):
    """Trainer class for weak form graph Koopman bilinear form (wGKBF) models."""
    
    def __init__(self, config_path: str, adj_mat: torch.Tensor):
        """Initialize wGKBF trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
            adj_mat: Adjacency matrix tensor for the graph structure
        """
        self.adj_mat = adj_mat
        super().__init__(config_path, weakGraphKBF)
    
    def _create_trajectory_manager(self):
        """Create TrajectoryManager with adjacency matrix for graph-based models."""
        from src.data.trajectory_manager import TrajectoryManager
        return TrajectoryManager(self.metadata, device=self.device, adj=self.adj_mat)
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        min_lr = 5e-5

        for trajectory in self.train_loader:
            batch = next(iter(trajectory)).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            # Extract states and controls
            states = batch.x
            controls = batch.u
            edge_index = batch.edge_index
            # Forward pass - specific to wMLP
            predictions = self.model(states, controls, edge_index)
            # Use weak form loss
            loss = weak_form_loss(states, predictions, self.metadata['weak_dyn_param'], self.criterion)
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
    
    def evaluate_rmse(self, split: str = 'test', plot: bool = False) -> float:
        """Calculate RMSE on a random trajectory from the specified split."""
        from src.losses.evaluation import prediction_rmse_graph

        dataset = getattr(self, f"{split}_set")
        trajectory = random.choice(dataset)
        return prediction_rmse_graph(
            self.model, trajectory, self.t, 
            self.metadata, self.model_name, plot=plot
        )