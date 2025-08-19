import logging
import numpy as np
import torch
import os
from typing import Callable, Optional, Tuple, Type

from dymad.data import TrajectoryManager
from dymad.models import KBF, DKBF
from dymad.utils import load_model

logger = logging.getLogger(__name__)

class SAInterface:
    """
    Interface for spectral analysis of KBF and DKBF models.

    It loads the model and data, sets up the necessary transformations,
    and provides methods to encode, decode, and apply observables.
    """

    def __init__(self, model_class: Type[torch.nn.Module], checkpoint_path: str):
        assert model_class in [KBF, DKBF], "Spectral Analysis is currently only implemented for KBF and DKBF."
        assert os.path.exists(checkpoint_path), "Checkpoint path does not exist."

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup paths
        _dir = os.path.dirname(checkpoint_path)
        _dir = '.' if _dir == '' else _dir
        os.makedirs(f'{_dir}/results', exist_ok=True)
        self.results_prefix = f'{_dir}/results'

        # Load model and data
        self._setup_data(checkpoint_path)

        mdl, _ = load_model(model_class, checkpoint_path)
        self._trans_x, self._trans_u, self.model = mdl

        self._setup_sa_terms()

        # Summary of information
        logger.info("SAInterface Initialized:")
        logger.info(self.model)
        logger.info(self.model.diagnostic_info())
        logger.info(f"Using device: {self.device}")

    def _setup_data(self, checkpoint_path) -> None:
        """Setup data loaders and datasets.
        
        Striped from TrainerBase.
        """
        self.metadata = torch.load(checkpoint_path, weights_only=False)['metadata']
        tm = TrajectoryManager(self.metadata, device=self.device)
        self.dataloaders, self.datasets, self.metadata = tm.process_all()
        self.train_loader, self.validation_loader, self.test_loader = self.dataloaders
        self.train_set, self.validation_set, self.test_set = self.datasets
        self.dtype = tm.dtype
        self.t = torch.tensor(tm.t[0])

    def _setup_sa_terms(self):
        P0, P1 = [], []
        for batch in self.train_loader:
            _P = self.model.encoder(batch, None)
            _P0, _P1 = _P[..., :-1, :], _P[..., 1:, :]
            _P0 = _P0.reshape(-1, _P0.shape[-1])
            _P1 = _P1.reshape(-1, _P1.shape[-1])
            P0.append(_P0)
            P1.append(_P1)
        self.P0 = torch.cat(P0, dim=0).cpu().numpy()
        self.P1 = torch.cat(P1, dim=0).cpu().numpy()

        self._Ninp = self.model.n_total_state_features
        self._Nout = self.model.koopman_dimension

    def get_weights(self) -> Tuple[np.ndarray]:
        if self.model.dynamics_net.mode == "full":
            return (self.model.dynamics_net.weight.data.cpu().numpy(), )
        else:
            U = self.model.dynamics_net.U.data.cpu().numpy()
            V = self.model.dynamics_net.V.data.cpu().numpy()
            return (U, V)

    def encode(self, X: np.ndarray, rng: Optional[list | None] = None) -> np.ndarray:
        """
        Encode new trajectory data to the observer space.
        """
        if rng is None:
            _X = self._trans_x.transform([X])[0]
            _X = torch.tensor(_X, dtype=self.dtype).to(self.device)
            _Z = self.model.encoder(_X, None).cpu().numpy()
            return _Z
        raise NotImplementedError("Encoding with a range is not implemented yet.")

    def decode(self, X: np.ndarray, rng: Optional[list | None] = None) -> np.ndarray:
        """
        Decode trajectory data from the observer space.
        """
        if rng is None:
            _X = torch.tensor(X, dtype=self.dtype).to(self.device)
            _Z = self.model.decoder(_X).cpu().numpy()
            _Z = self._trans_x.transform([_Z])[0]
            return _Z
        raise NotImplementedError("Decoding with a range is not implemented yet.")

    def apply_obs(self, fobs: Callable) -> np.ndarray:
        """
        Apply a generic observable to the data.

        Args:
            fobs (Callable): Observable function. It should accept a 2D array input with each row as one step.
                             The output should be a 1D array, whose ith entry corresponds to the ith step.
        """
        F = []
        for batch in self.train_loader:
            B = batch.cpu().numpy()
            B = B.reshape(-1, B.shape[-1])
            F.append(fobs(B))
        return np.vstack(F)
