import logging
import numpy as np
import random
import time
import torch
import os
from typing import Dict, List, Tuple, Type

from dymad.data import TrajectoryManager, TrajectoryManagerGraph
from dymad.losses import prediction_rmse
from dymad.utils import load_checkpoint, load_config, plot_hist, save_checkpoint

logger = logging.getLogger(__name__)

class TrainerBase:
    """Base trainer class for dynamical system models.

    Args:
        config_path (str): Path to the YAML configuration file
        model_class (Type[torch.nn.Module]): Class of the model to train
    """
    def __init__(self, config_path: str, model_class: Type[torch.nn.Module], config_mod: Dict = None):
        self.config = load_config(config_path, config_mod)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.config['model']['name']
        self.model_class = model_class

        if self.config['training'].get('Tolerance_sweeps') is not None:
            self.convergence_tolerance = float(self.config['training']['Tolerance_sweeps'][-1])
        else:
            self.convergence_tolerance = None

        # Setup paths
        os.makedirs('./checkpoints', exist_ok=True)
        self.checkpoint_path = f'./checkpoints/{self.model_name}_checkpoint.pt'
        self.best_model_path = f'./{self.model_name}.pt'
        os.makedirs('./results', exist_ok=True)
        self.results_prefix = './results'
        # Initialize metadata
        self.metadata = self._init_metadata()
        # Setup data
        self._setup_data()

        # Setup model and training components
        self._setup_model()
        # Training history
        self.start_epoch, self.best_loss, self.hist, self.rmse, _ = self._load_checkpoint()

        # Summary of information
        logger.info("Trainer Initialized:")
        logger.info(f"Model name: {self.model_name}")
        logger.info(self.model)
        logger.info(self.model.diagnostic_info())
        logger.info("Optimization settings:")
        logger.info(self.optimizer)
        logger.info(self.criterion)
        logger.info(f"LR decay: {self.schedulers[0].gamma}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Double precision: {self.config['data']['double_precision']}")
        logger.info(f"Epochs: {self.config['training']['n_epochs']}, Save interval: {self.config['training']['save_interval']}")

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
        if self.model_class.GRAPH:
            return TrajectoryManagerGraph(self.metadata, device=self.device)
        else:
            return TrajectoryManager(self.metadata, device=self.device)

    def _setup_model(self) -> None:
        """Setup model, optimizer, scheduler and criterion."""
        self.model = self.model_class(self.config['model'], self.metadata).to(self.device)
        if self.config['data']['double_precision']:
            self.model = self.model.double()

        # By default there is only one scheduler.
        # There might be more, e.g., in NODETrainer with sweep scheduler.
        _lr = float(self.config['training'].get('learning_rate', 1e-3))
        _gm = float(self.config['training'].get('decay_rate', 0.999))
        self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=_lr)
        self.schedulers = [torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=_gm)]
        self.criterion  = torch.nn.MSELoss(reduction='mean')

    def _load_checkpoint(self) -> Tuple[int, float, List, Dict]:
        """Load checkpoint if it exists."""
        return load_checkpoint(
            self.model, self.optimizer, self.schedulers,
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
        return prediction_rmse

    def get_evaluation_dataset(self, split: str):
        """
        Get the dataset for evaluation. Override in subclasses if needed.

        Args:
            split (str): Dataset split to use ('train', 'validation', 'test')

        Returns:
            List[torch.Tensor]: Dataset for evaluation
        """
        return getattr(self, f"{split}_set")

    def evaluate_rmse(self, split: str = 'test', plot: bool = False, evaluate_all: bool = False) -> float:
        """
        Calculate RMSE on trajectory(ies) from the specified split.

        Args:
            split (str): Dataset split to use ('train', 'validation', 'test')
            plot (bool): Whether to plot the results (only works when evaluate_all=False)
            evaluate_all (bool):

                - If True, evaluate all trajectories and return mean RMSE.
                - If False, evaluate a single random trajectory.

        Returns:
            float: RMSE value (mean RMSE if evaluate_all=True)
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
            prediction_rmse_func(self.model, trajectory, self.t, self.metadata, self.model_name,
                                 plot=plot, prefix=self.results_prefix)
            for trajectory in dataset
        ]

        return sum(rmse_values) / len(rmse_values)

    def save_if_best(self, val_loss: float, epoch: int) -> bool:
        """Save model if validation loss improved."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.convergence_epoch = epoch+1
            save_checkpoint(
                self.model, self.optimizer, self.schedulers,
                epoch, self.best_loss, self.hist, self.rmse, self.metadata,
                self.best_model_path
            )
            logger.info(f"Best model saved at epoch {epoch+1} with validation loss {self.best_loss:.4e}")
            return True
        return False

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        save_checkpoint(
            self.model, self.optimizer, self.schedulers,
            epoch, self.best_loss, self.hist, self.rmse, self.metadata,
            self.checkpoint_path
        )

    def train(self) -> None:
        """Run full training loop."""

        n_epochs = self.config['training']['n_epochs']
        save_interval = self.config['training']['save_interval']

        self.convergence_epoch = None
        self.epoch_times = []

        overall_start_time = time.time()
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            # Training and evaluation
            # Only timing the train and validation phases
            # since test loss is only for reference
            epoch_start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.evaluate(self.validation_loader)

            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            test_loss = self.evaluate(self.test_loader)

            # Record history
            self.hist.append([epoch, train_loss, val_loss, test_loss])

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
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)

                # Plot loss curves
                plot_hist(self.hist, epoch+1, self.model_name, prefix=self.results_prefix)

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

            if self.convergence_tolerance is not None:
                if val_loss < self.convergence_tolerance:
                    logger.info(f"Convergence reached at epoch {epoch+1} "
                                f"with validation loss {val_loss:.4e}")
                    break

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
