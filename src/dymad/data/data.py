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
        # The array is assumed to be of shape (batch_size, n_steps, n_features)
        # unfold produces a tensor of shape (batch_size, n_window, n_features, window)
        # merge the first two dimensions and permute the last two gives (batch_size*n_window, window, n_features)
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
    n_nodes: int = 0
    """n_nodes (int): Number of nodes in the graph structure."""

    def __post_init__(self):
        self.n_nodes = self.edge_index.max().item() + 1
        if self.x is not None:
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1)
        if self.u is not None:
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1)

    @property
    def xg(self) -> torch.Tensor:
        """
        Get the state tensor with shape (batch_size, n_steps, n_nodes, n_features).
        """
        return self.x.reshape(*self.x_reshape)

    @property
    def ug(self) -> Union[torch.Tensor, None]:
        """
        Get the control tensor with shape (batch_size, n_steps, n_nodes, n_controls).
        Returns None if control tensor is not present.
        """
        if self.u is not None:
            return self.u.reshape(*self.u_reshape)
        return None

    def g(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reshape a tensor to have shape (batch_size, n_steps, n_nodes, -1).

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, n_steps, n_nodes * features).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, n_steps, n_nodes, features).
        """
        out_shape = z.shape[:-1] + (self.n_nodes, -1)
        return z.reshape(*out_shape)
    
    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        The reverse of g()
        """
        out_shape = z.shape[:-2] + (-1,)
        return z.reshape(*out_shape)

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

        The operation follows PyGData, that assembles the graphs of each sample
        into a single large graph, so that the subsequent GNN evaluation operates
        on a single graph to maximize parallelism.

        Args:
            batch_list (List[DynGeoData]): List of DynGeoData instances to collate.

        Returns:
            DynGeoData: A single DynGeoData instance with stacked state and control tensors.
        """
        xs = torch.concatenate([b.x for b in batch_list], dim=-1).unsqueeze(0)
        if batch_list[0].u is not None:
            us = torch.concatenate([b.u for b in batch_list], dim=-1).unsqueeze(0)
        else:
            us = None

        n_nodes = [0] + [b.n_nodes for b in batch_list[:-1]]
        offset = torch.tensor(n_nodes).cumsum(dim=0)
        edge_index = torch.concatenate([
            b.edge_index + offset[i] for i, b in enumerate(batch_list)],
            dim=-1).unsqueeze(0)

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
            DynGeoDataImpl: A new DynGeoDataImpl instance with unfolded data.
        """
        x_tmp = self.x.unfold(1, window, stride)
        n_window = x_tmp.size(1)
        x_unfolded = x_tmp.reshape(-1, self.x.size(-1), window).permute(0, 2, 1)
        u_unfolded = self.u.unfold(1, window, stride).reshape(-1, self.u.size(-1), window).permute(0, 2, 1) if self.u is not None else None
        # Repeat edge_index along the window dimension and reshape
        e_unfolded = self.edge_index.unsqueeze(1).repeat(1, n_window, 1, 1).reshape(-1, self.edge_index.size(1), self.edge_index.size(2))
        return DynGeoDataImpl(x_unfolded, u_unfolded, e_unfolded)
