from abc import ABC
import torch
import torch.nn as nn
from typing import Tuple, Union

from dymad.data import DynData, DynGeoData

Data = Union[DynData, DynGeoData]

class ModelBase(nn.Module, ABC):
    r"""
    Base class for dynamic models.

    Notation:

    - x: Physical state/observation space.
    - u: Control input.
    - z: Embedding (latent) space.

    Discrete-time model:

    - z_k = encoder(x_k, u_k)
    - z_{k+1} = dynamics(z_k, u_k)
    - x_{k+1} = decoder(z_{k+1})

    Continuous-time model:

    - z = encoder(x, u)
    - \dot{z} = dynamics(z, u)
    - x = decoder(z)

    Linear training assumes:

    - linear_targets = dynamics = W @ linear_features(z, u)
    - and fits W only
    """
    GRAPH = None   # True for graph compatible models
    CONT  = None   # True for continuous-time models, otherwise discrete-time

    def __init__(self):
        super(ModelBase, self).__init__()

    def diagnostic_info(self) -> str:
        """
        Return diagnostic information about the model.

        Returns:
            str: String with model details
        """
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n"

    def encoder(self, w: Data) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def decoder(self, z: torch.Tensor, w: Data) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def dynamics(self, z: torch.Tensor, w: Data) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def predict(self, x0: torch.Tensor, us: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")

    def forward(self, w: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")

    def linear_features(self, w: Data) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    def set_linear_weights(self, W: torch.Tensor) -> None:
        raise NotImplementedError("This is the base class.")