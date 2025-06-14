import yaml, logging, os, torch, random
from typing import Dict, List, Tuple, Type

logger = logging.getLogger(__name__)

class TrainerBase:
    """Base trainer class for dynamical system models."""
    
    def __init__(self, config_path: str, model_class: Type[torch.nn.Module]):
        """
        Initialize trainer with configuration.
        Args:
            config_path: Path to the YAML configuration file
            model_class: Class of the model to train
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.config['model']['name']
        self.model_class = model_class
        
        # Setup paths
        os.makedirs('./checkpoints', exist_ok=True)
        self.checkpoint_path = f'./checkpoints/{self.model_name}_checkpoint.pt'
        self.best_model_path = f'./{self.model_name}.pt'
        # Initialize metadata
        self.metadata = self._init_metadata()
        # Setup data
        self._setup_data()

        # Setup model and training components
        self._setup_model()
        # Training history
        self.start_epoch, self.best_loss, self.hist, _ = self._load_checkpoint()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Double precision: {self.config['data']['double_precision']}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _init_metadata(self) -> Dict:
        """Initialize metadata from config or checkpoint."""
        if os.path.exists(self.checkpoint_path) and self.config['training']['load_checkpoint']:
            logger.info(f"Checkpoint found at {self.checkpoint_path}, overriding the yaml config.")
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            return checkpoint['metadata']
        else:
            logger.info(f"No checkpoint found at {self.checkpoint_path}, using the yaml config.")
            return {'config': self.config}
    
    def _setup_data(self) -> None:
        """Setup data loaders and datasets."""
        tm = self._create_trajectory_manager()
        self.dataloaders, self.datasets, self.metadata = tm.process_all()
        self.train_loader, self.validation_loader, self.test_loader = self.dataloaders
        self.train_set, self.validation_set, self.test_set = self.datasets
        self.t = torch.tensor(tm.t[0])  # TODO: check trajectory of different lengths
    def _create_trajectory_manager(self):
        """Create and return a TrajectoryManager instance.
        Override this method in subclasses to customize TrajectoryManager creation."""
        from src.data.trajectory_manager import TrajectoryManager
        return TrajectoryManager(self.metadata, device=self.device)
    
    def _setup_model(self) -> None:
        """Setup model, optimizer, scheduler and criterion."""
        self.model = self.model_class(self.config['model'], self.metadata).to(self.device)
        if self.config['data']['double_precision']:
            self.model = self.model.double()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        logger.info("Model:")
        logger.info(self.model)
    
    def _load_checkpoint(self) -> Tuple[int, float, List, Dict]:
        """Load checkpoint if it exists."""
        from src.utils.checkpoint import load_checkpoint
        return load_checkpoint(
            self.model, self.optimizer, self.scheduler, 
            self.checkpoint_path, self.config['training']['load_checkpoint']
        )
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate the model on the provided dataloader."""
        raise NotImplementedError("Subclasses must implement evaluate")
    
    def get_prediction_rmse_func(self):
        """
        Return the appropriate prediction RMSE function for this trainer.
        Override in subclasses to use different evaluation functions.
        """
        from src.losses.evaluation import prediction_rmse
        return prediction_rmse
    
    def get_evaluation_dataset(self, split: str):
        """
        Get the dataset for evaluation. Override in subclasses if needed.
        
        Args:
            split: Dataset split to use ('train', 'validation', 'test')
            
        Returns:
            Dataset for evaluation
        """
        return getattr(self, f"{split}_set")
    
    def evaluate_rmse(self, split: str = 'test', plot: bool = False, evaluate_all: bool = False) -> float:
        """
        Calculate RMSE on trajectory(ies) from the specified split.
        
        Args:
            split: Dataset split to use ('train', 'validation', 'test')
            plot: Whether to plot the results (only works when evaluate_all=False)
            evaluate_all: If True, evaluate all trajectories and return mean RMSE.
                         If False, evaluate a single random trajectory.
        
        Returns:
            RMSE value (mean RMSE if evaluate_all=True)
        """
        
        # Get the appropriate prediction function and dataset
        prediction_rmse_func = self.get_prediction_rmse_func()
        full_dataset = self.get_evaluation_dataset(split)
        
        if evaluate_all:
            plot = False
            dataset = full_dataset
        else:
            dataset = [random.choice(full_dataset)]
        
        rmse_values = [
            prediction_rmse_func(self.model, trajectory, self.t, self.metadata, self.model_name, plot=plot)
            for trajectory in dataset
        ]
        
        return sum(rmse_values) / len(rmse_values)
    
    def save_if_best(self, val_loss: float, epoch: int) -> bool:
        """Save model if validation loss improved."""
        from src.utils.checkpoint import save_checkpoint
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(
                self.model, self.optimizer, self.scheduler, 
                epoch, self.best_loss, self.hist, self.metadata, 
                self.best_model_path
            )
            logger.info(f"Best model saved at epoch {epoch+1} with validation loss {self.best_loss:.4e}")
            return True
        return False
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        from src.utils.checkpoint import save_checkpoint
        save_checkpoint(
            self.model, self.optimizer, self.scheduler, 
            epoch, self.best_loss, self.hist, self.metadata, 
            self.checkpoint_path
        )
    
    def train(self) -> None:
        """Run full training loop."""
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
            logger.info(
                f"Epoch {epoch+1}/{n_epochs}, "
                f"Train Loss: {train_loss:.4e}, "
                f"Validation Loss: {val_loss:.4e}, "
                f"Test Loss: {test_loss:.4e}"
            )
            
            # Plotting
            plot_hist(self.hist, epoch+1, self.model_name)
            
            # Save best model
            self.save_if_best(val_loss, epoch)

            # Periodic checkpoint and evaluation
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)
                
                # Evaluate RMSE on random trajectories
                rmse_train = self.evaluate_rmse('train', plot=False)
                rmse_val = self.evaluate_rmse('validation', plot=False)
                rmse_test = self.evaluate_rmse('test', plot=True)
            
                logger.info(
                    f"Prediction RMSE - "
                    f"Train: {rmse_train:.4e}, "
                    f"Validation: {rmse_val:.4e}, "
                    f"Test: {rmse_test:.4e}"
                )