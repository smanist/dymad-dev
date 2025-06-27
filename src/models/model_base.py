from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple

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

    def diagnostic_info(self) -> str:
        return f"Model parameters: {sum(p.numel() for p in self.parameters())}\n"

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
