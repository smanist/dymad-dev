import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod

class ModelBase(nn.Module, ABC):
    """
    Base class for dynamic models.

    Notation:
      x: Physical state space.
      z: Embedding (latent) space.

    Discrete-time model:
      z_k = encoder(x_k)
      z_k+1 = dynamics(z_k, u_k)
      x_k+1 = decoder(z_k+1)

    Continuous-time model:
      z = encoder(x)
      z_dot = dynamics(z, u)
      x = decoder(z)
    """
    def __init__(self):
        super(ModelBase, self).__init__()

    @abstractmethod
    def init_params(self):
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def features(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is the base class.")

    @abstractmethod
    def predict(self, x0: torch.Tensor, us: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This is the base class.")
    
    @staticmethod
    def build_mlp(input_dimension: int, latent_dimension: int, output_dimension: int, num_layers: int) -> nn.Sequential:
        if num_layers == 1:
            return nn.Sequential(
                nn.Linear(input_dimension, output_dimension),
                nn.PReLU()
            )
        layers = [nn.Linear(input_dimension, latent_dimension), nn.PReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(latent_dimension, latent_dimension), nn.PReLU()])
        layers.append(nn.Linear(latent_dimension, output_dimension))
        layers.append(nn.PReLU())
        return nn.Sequential(*layers)    
