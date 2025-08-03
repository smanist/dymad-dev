"""
The class names have a `Impl` suffix to avoid conflicts with the `DynData` and `DynGeoData` classes
defined in `dymad.data.__init__.py`. This avoids confusion in Sphinx in documentation generation.
"""

from dataclasses import dataclass
import torch
from typing import List, Union

@dataclass
class DynDataImpl:
    """
    Data structure for dynamic data, containing state and control tensors.
    """
    x: torch.Tensor
    """x (torch.Tensor): State tensor of shape (batch_size, n_steps, n_features)."""

    u: Union[torch.Tensor, None]
    """u (torch.Tensor): Control tensor of shape (batch_size, n_steps, n_controls)."""

    def to(self, device: torch.device, non_blocking: bool = False) -> "DynDataImpl":
        """
        Move the state and control tensors to a different device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): If True, the operation will be non-blocking.

        Returns:
            DynData: A DynData instance with tensors on the target device.
        """
        self.x = self.x.to(device, non_blocking=non_blocking)
        if self.u is not None:
            self.u = self.u.to(device, non_blocking=non_blocking)
        return self

    @classmethod
    def collate(cls, batch_list: List["DynDataImpl"]) -> "DynDataImpl":
        """
        Collate a list of DynData instances into a single DynData instance.
        Needed by DataLoader to stack state and control tensors.

        Args:
            batch_list (List[DynData]): List of DynData instances to collate.

        Returns:
            DynData: A single DynData instance with stacked state and control tensors.
        """
        xs = torch.stack([b.x for b in batch_list], dim=0)
        if batch_list[0].u is not None:
            us = torch.stack([b.u for b in batch_list], dim=0)
        else:
            us = None
        return DynDataImpl(xs, us)

    def truncate(self, num_step):
        return DynDataImpl(self.x[:, :num_step, :],
                           self.u[:, :num_step, :] if self.u is not None else None)
    
    def unfold(self, window: int, stride: int) -> "DynDataImpl":
        """
        Unfold the data into overlapping windows.

        Args:
            window (int): Size of the sliding window.
            stride (int): Step size for the sliding window.

        Returns:
            DynDataImpl: A new DynDataImpl instance with unfolded data.
        """
        x_unfolded = self.x.unfold(1, window, stride).reshape(-1, self.x.size(-1), window).permute(0, 2, 1)
        u_unfolded = self.u.unfold(1, window, stride).reshape(-1, self.u.size(-1), window).permute(0, 2, 1) if self.u is not None else None
        return DynDataImpl(x_unfolded, u_unfolded)

@dataclass
class DynGeoDataImpl:
    """
    Data structure for dynamic geometric data, containing state and control tensors,
    and topological information.
    """
    x: torch.Tensor
    """x (torch.Tensor): State tensor of shape (batch_size, n_steps, n_features)."""
    u: Union[torch.Tensor, None]
    """u (torch.Tensor): Control tensor of shape (batch_size, n_steps, n_controls)."""
    edge_index: torch.Tensor
    """edge_index (torch.Tensor): Edge index tensor for graph structure, shape (2, n_edges)."""

    def to(self, device: torch.device, non_blocking: bool = False) -> "DynGeoDataImpl":
        """
        Move the data to a different device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): If True, the operation will be non-blocking.

        Returns:
            DynGeoData: A DynGeoData instance with tensors on the target device.
        """
        self.x = self.x.to(device, non_blocking=non_blocking)
        if self.u is not None:
            self.u = self.u.to(device, non_blocking=non_blocking)
        self.edge_index = self.edge_index.to(device, non_blocking=non_blocking)
        return self

    @classmethod
    def collate(cls, batch_list: List["DynGeoDataImpl"]) -> "DynGeoDataImpl":
        """
        Collate a list of DynGeoData instances into a single DynGeoData instance.
        Needed by DataLoader to stack state and control tensors.

        Args:
            batch_list (List[DynGeoData]): List of DynGeoData instances to collate.

        Returns:
            DynGeoData: A single DynGeoData instance with stacked state and control tensors.
        """
        xs = torch.stack([b.x for b in batch_list], dim=0)
        if batch_list[0].u is not None:
            us = torch.stack([b.u for b in batch_list], dim=0)
        else:
            us = None
        edge_index = torch.stack([b.edge_index for b in batch_list], dim=0)
        return DynGeoDataImpl(xs, us, edge_index)

    def truncate(self, num_step):
        return DynGeoDataImpl(self.x[:, :num_step, :],
                              self.u[:, :num_step, :] if self.u is not None else None,
                              self.edge_index)

    def unfold(self, window: int, stride: int) -> "DynGeoDataImpl":
        """
        Unfold the data into overlapping windows.

        Args:
            window (int): Size of the sliding window.
            stride (int): Step size for the sliding window.

        Returns:
            DynDataImpl: A new DynDataImpl instance with unfolded data.
        """
        x_unfolded = self.x.unfold(1, window, stride).reshape(-1, self.x.size(-1), window).permute(0, 2, 1)
        u_unfolded = self.u.unfold(1, window, stride).reshape(-1, self.u.size(-1), window).permute(0, 2, 1) if self.u is not None else None
        return DynGeoDataImpl(x_unfolded, u_unfolded, self.edge_index)
