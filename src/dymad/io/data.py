from dataclasses import dataclass, field
import torch
from typing import List, Optional, Union, Tuple

def _ensure_batches(tensor: torch.Tensor, base_dim: int, offset: int = 0) -> torch.Tensor:
    if tensor is None:
        return None
    _dim = base_dim - offset
    if tensor.ndim == _dim + 1:
        # (n_steps, ...)
        return tensor.reshape(1, *tensor.shape)
    elif tensor.ndim == _dim + 2:
        # (batch_size, n_steps, ...)
        return tensor
    else:
        raise ValueError(f"Invalid tensor shape for DynData: {tensor.shape}. Expected shape with {base_dim} or {base_dim+1} dimensions.")

def _determine_sizes(lst: List[torch.Tensor]) -> Tuple[int, int]:
    batch_size, n_steps = None, None
    for _d in lst:
        if _d is not None:
            if _d.ndim == 2:
                batch_size = 1
                n_steps = _d.shape[0]
            else:
                batch_size = _d.shape[0]
                n_steps = _d.shape[1]
            break
    return batch_size, n_steps

def _process_graph_format_single(lst: Union[List[torch.Tensor], torch.Tensor, None], base_dim) -> torch.Tensor:
    """
    Considering the following cases:

    - None: return None
    - Single trajectory:

        - List of tensors of shape (n_edges, ...) -> convert to NestedTensor
        - 3D tensor of shape (n_steps, n_edges, ...) -> convert to NestedTensor
    """
    if lst is None:
        return None

    if isinstance(lst, list):
        assert isinstance(lst[0], torch.Tensor), "List items must be torch.Tensor"
        assert lst[0].ndim == base_dim, f"List items must have {base_dim} dimensions"
        return torch.nested.nested_tensor(lst, dtype=lst[0].dtype, layout=torch.jagged)

    if isinstance(lst, torch.Tensor):
        if lst.ndim != base_dim + 1:
            raise ValueError(f"Invalid tensor shape for graph data: {lst.shape}. Expected shape with base dim {base_dim} or {base_dim+1}.")
        return torch.nested.nested_tensor(lst.unbind(), dtype=lst.dtype, layout=torch.jagged)

    raise ValueError(f"Invalid type for graph data: {type(lst)}. Expected list or torch.Tensor.")

def _collate_nested_tensor(lst: Union[List[torch.Tensor], torch.Tensor], offset: Union[List[int], bool] = False) -> torch.Tensor:
    """
    lst is (batch_size,) list of tensors with shape (n_steps, n_edges, ...),
    where n_edges can be jagged.  The method collates the edge dimension over the batch,
    resulting in shape (n_steps, n_total_edges, ...).

    Offset is need for edge indices to shift the node indices accordingly.
    """
    if offset is not False:  # But can be a given list
        # Offset is needed
        if offset is True:
            # Need to compute offsets
            n_nodes = [0] + [l.values().max().item() + 1 for l in lst[:-1]]
            offset = torch.tensor(n_nodes).cumsum(dim=0)
        lst = [l + offset[i] for i, l in enumerate(lst)]

    collated = []
    for step_items in zip(*lst):  # step_items is (batch_size,) tuple of (n_edges, ...)
        collated.append(torch.concatenate(step_items, dim=0))
    return torch.nested.nested_tensor(collated, layout=torch.jagged, dtype=lst[0].dtype)

def _ensure_one_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return None
    if tensor.ndim == 3:
        # (batch_size, n_steps, ...)
        if tensor.size(0) != 1:
            # Swap batch and time dimensions, then flatten
            tensor = tensor.permute(1, 0, 2)
            return tensor.reshape(1, tensor.shape[0], -1)
        # Already batch size 1
        return tensor
    elif tensor.ndim == 2:
        # (n_steps, ...)
        return tensor.unsqueeze(0)
    else:
        raise ValueError(f"Invalid tensor shape for DynData in graph mode: {tensor.shape}. " \
                         f"Expected shape with 2D or 3D.")

def _ensure_graph_format(
        lst: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor, None],
        base_dim, need_offset: bool = False) -> torch.Tensor:
    """
    Considering the following cases:

    - None or Single trajectory: back to _process_graph_format_single
    - Batch of trajectories -> collate into NestedTensor

        - 4D tensor of shape (batch_size, n_steps, n_edges, ...)
        - List of tensors of shape (n_steps, n_edges, ...)
        - List of lists of tensors of shape (n_edges, ...)
    """
    # Batch cases
    if isinstance(lst, torch.Tensor) and lst.ndim == base_dim + 2:
        return _collate_nested_tensor(lst, need_offset), lst.size(0)

    if isinstance(lst, list):
        if isinstance(lst[0], torch.Tensor) and lst[0].ndim == base_dim + 1:
            return _collate_nested_tensor(lst, need_offset), len(lst)
        elif isinstance(lst[0], list) and isinstance(lst[0][0], torch.Tensor) and lst[0][0].ndim == base_dim:
            tmp = [_process_graph_format_single(sublist, base_dim) for sublist in lst]
            return _collate_nested_tensor(tmp, need_offset), len(lst)

    # Else:
    return _process_graph_format_single(lst, base_dim), 1

def _slice(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """
    Slice a NestedTensor along the first dimension (time steps).

    Args:
        tensor (torch.Tensor): Input NestedTensor of shape (n_steps, ...).
        start (int): Start index for slicing.
        end (int): End index for slicing.

    Returns:
        torch.Tensor: Sliced NestedTensor of shape (end - start, ...).
    """
    offsets = tensor.offsets()
    offset_0 = offsets[start]
    offset_1 = offsets[end]
    sliced_values = tensor.values()[offset_0:offset_1]
    return torch.nested.nested_tensor_from_jagged(
        sliced_values,
        offsets[start:end + 1] - offsets[start])

def _unfold(tensor: torch.Tensor, window: int, stride: int, offset: int = 0) -> torch.Tensor:
    """
    Unfold a NestedTensor along the first dimension (time steps).

    In this specialized unfolding, the graph size is expanded by n_windows times.

    Args:
        tensor (torch.Tensor): Input NestedTensor of shape (n_steps, n_edges, ...).
        window (int): Size of the sliding window.
        stride (int): Step size for the sliding window.
        offset (int): Offset to be added to edge indices.

    Returns:
        torch.Tensor: Unfolded NestedTensor of shape (window, n_windows x n_edges, ...).
    """
    if tensor is None:
        return None

    n_window = (tensor.size(0) - window) // stride + 1
    data = tensor.unbind()
    buffer = []
    for i in range(n_window):
        # Loop over windows, the ith window is the ith expansion of the graph
        # so the indices need to add i*offset
        I, J = i*stride, i*stride + window
        buffer.append([data[j]+i*offset for j in range(I, J)])

    # Same operation as collate
    collated = []
    for step_items in zip(*buffer):
        collated.append(torch.concatenate(step_items, dim=0))
    return torch.nested.nested_tensor(collated, layout=torch.jagged, dtype=buffer[0][0].dtype)

@dataclass
class DynData:
    """
    Data structure for time series data.

    Operates in normal mode or graph mode depending on the presence of edge_index.

    - In normal mode, data shape is (batch_size, n_steps, ...).  If the batch_size dimension is missing,
      it is assumed to be 1.
    - In graph mode, batch_size is always 1, so

        - Regular data shape is (1, n_steps, ...),
        - Graph data shape is (n_steps, n_edges, ...)

    The reason is to always aggregate graphs as much as possible for GNN evaluation efficiency.
    As a result, in the two modes, the collate and unfold methods behave differently.

    In addition, in graph mode, the number of nodes are assumed to be the same across
    all time steps (for now).  But the number of edges can vary per time step; therefore,
    the edge-related data members are stored as NestedTensors.
    """
    # Generic data
    t: Optional[torch.Tensor] = None
    """t (torch.Tensor): Time tensor of shape (batch_size, n_steps)."""

    x: Optional[torch.Tensor] = None
    """
    x (torch.Tensor): Feature tensor of shape (batch_size, n_steps, n_features).
    The only data member that is always required.
    """

    y: Optional[torch.Tensor] = None
    """
    y (torch.Tensor): Auxiliary tensor of shape (batch_size, n_steps, n_aux).
    Additional data that may be used for supervised learning, e.g., additional observations.
    """

    u: Optional[torch.Tensor] = None
    """u (torch.Tensor): Control tensor of shape (batch_size, n_steps, n_controls)."""

    p: Optional[torch.Tensor] = None
    """
    p (torch.Tensor): Parameter tensor of shape (batch_size, n_parameters).
    Meant for constants that do not vary with time.
    If there are intermittent step changes in parameters, they should be treated as controls in u.
    """

    # Graph-only
    ei: Optional[torch.Tensor] = None
    """
    edge_index (torch.Tensor): Edge index tensor for graph structure, shape (n_steps, n_edges, 2).
    Using nested tensor if n_edges varies with time.
    """

    ew: Optional[torch.Tensor] = None
    """
    edge_weight (torch.Tensor): Edge weight tensor for graph structure, shape (n_edges, n_steps).
    Using nested tensor if n_edges varies with time.
    """

    ea: Optional[torch.Tensor] = None
    """
    edge_attr (torch.Tensor): Edge attribute tensor for graph structure, shape (n_edges, n_steps, n_edge_features).
    Using nested tensor if n_edges varies with time.
    """

    n_nodes: int = 0
    """n_nodes (int): Number of nodes in the graph structure."""

    batch_size: Optional[int] = None
    """
    batch_size (int): True batch size.
    In graph mode, the apparent batch size is 1, but batch_size stores the true batch size.
    """

    n_steps: Optional[int] = None
    """n_steps (int): Number of time steps."""

    # Other data
    meta: List[dict] = field(default_factory=list)
    """meta (List[dict]): Metadata for each time series."""

    _has_graph: bool = False
    """_has_graph (bool): Whether the data has graph structure."""

    def __post_init__(self):
        if self.ei is not None:
            # There is graph structure
            self._has_graph = True
            # Ensure batch size 1
            ## Assuming inputs either follow normal mode, (batch_size, n_steps, ...) {more natural to user}
            ## or graph mode, (1, n_steps, ...) {typically from collate}
            if self.t is not None:
                if self.t.ndim == 1:
                    # Assuming input is (n_steps,) (natural to user for single traj)
                    self.t = self.t.unsqueeze(0)
                elif self.t.ndim == 2:
                    # Assuming input is (batch_size, n_steps) (natural to user for multiple trajs)
                    self.t = self.t.transpose(1, 0).unsqueeze(0)
                elif self.t.ndim == 3:
                    # (1, n_steps, batch_size)
                    # This typically happens after collate
                    assert self.t.size(0) == 1, "In graph mode, batch size must be 1."
                else:
                    raise ValueError(f"Invalid tensor shape for DynData.t in graph mode: {self.t.shape}.")
            self.x = _ensure_one_batch(self.x)
            self.y = _ensure_one_batch(self.y)
            self.u = _ensure_one_batch(self.u)
            self.p = self.p.reshape(1, -1) if self.p is not None else None
            # Ensure NestedTensor
            self.ei, _batch_size = _ensure_graph_format(self.ei, base_dim=2, need_offset=True)
            self.ew, _ = _ensure_graph_format(self.ew, base_dim=1, need_offset=False)
            self.ea, _ = _ensure_graph_format(self.ea, base_dim=2, need_offset=False)

            if self.batch_size is None:
                self.batch_size = _batch_size
                # self.batch_size might have been set in collate
            _, self.n_steps = _determine_sizes([self.x, self.u])

            self.n_nodes = self.ei.values().max().item() + 1
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1) if self.x is not None else None
            self.y_reshape = self.y.shape[:-1] + (self.n_nodes, -1) if self.y is not None else None
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1) if self.u is not None else None
            self.p_reshape = self.p.shape[:-1] + (self.n_nodes, -1) if self.p is not None else None
        else:
            self._has_graph = False

            self.t = _ensure_batches(self.t, base_dim=0)
            self.x = _ensure_batches(self.x, base_dim=1)
            self.y = _ensure_batches(self.y, base_dim=1)
            self.u = _ensure_batches(self.u, base_dim=1)
            self.p = _ensure_batches(self.p, base_dim=1, offset=1)

            self.batch_size, self.n_steps = _determine_sizes([self.x, self.u])

    def to(self, device: torch.device, non_blocking: bool = False) -> "DynData":
        """
        Move data tensors to a different device.

        Args:
            device (torch.device): The target device.
            non_blocking (bool, optional): If True, the operation will be non-blocking.

        Returns:
            DynData: A DynData instance with tensors on the target device.
        """
        self.t = self.t.to(device, non_blocking=non_blocking) if self.t is not None else None
        self.x = self.x.to(device, non_blocking=non_blocking) if self.x is not None else None
        self.y = self.y.to(device, non_blocking=non_blocking) if self.y is not None else None
        self.u = self.u.to(device, non_blocking=non_blocking) if self.u is not None else None
        self.p = self.p.to(device, non_blocking=non_blocking) if self.p is not None else None
        if self._has_graph:
            self.ei = self.ei.to(device, non_blocking=non_blocking) # self.ei is always not None in graph mode
            self.ew = self.ew.to(device, non_blocking=non_blocking) if self.ew is not None else None
            self.ea = self.ea.to(device, non_blocking=non_blocking) if self.ea is not None else None
        return self

    @classmethod
    def collate(cls, batch_list: List["DynData"]) -> "DynData":
        """
        Collate a list of DynData instances into a single DynData instance.
        Needed by DataLoader to stack data tensors.

        The graph version follows PyGData, that assembles the graphs of each sample
        into a single large graph, so that the subsequent GNN evaluation operates
        on a single graph to maximize parallelism.

        Args:
            batch_list (List[DynData]): List of DynData instances to collate.

        Returns:
            DynData: A single DynData instance with stacked data tensors.
        """
        if len(batch_list) == 1:
            return batch_list[0]    # No need to collate

        ms = []
        for b in batch_list:
            ms += b.meta

        if batch_list[0]._has_graph:
            ts = torch.concatenate([b.t for b in batch_list], dim=0).transpose(0, 1).unsqueeze(0) if batch_list[0].t is not None else None
            xs = torch.concatenate([b.x for b in batch_list], dim=-1) if batch_list[0].x is not None else None
            ys = torch.concatenate([b.y for b in batch_list], dim=-1) if batch_list[0].y is not None else None
            us = torch.concatenate([b.u for b in batch_list], dim=-1) if batch_list[0].u is not None else None
            ps = torch.concatenate([b.p for b in batch_list], dim=-1) if batch_list[0].p is not None else None

            n_nodes = [0] + [b.n_nodes for b in batch_list[:-1]]
            offset = torch.tensor(n_nodes).cumsum(dim=0)

            ei = _collate_nested_tensor([b.ei for b in batch_list], offset)
            ew = _collate_nested_tensor([b.ew for b in batch_list], False) if batch_list[0].ew is not None else None
            ea = _collate_nested_tensor([b.ea for b in batch_list], False) if batch_list[0].ea is not None else None

            # Collate already aggregates graphs, so batch_size is carried over to give true batch size
            return DynData(t=ts, x=xs, y=ys, u=us, p=ps, ei=ei, ew=ew, ea=ea, meta=ms, batch_size=len(batch_list))

        ts = torch.concatenate([b.t for b in batch_list], dim=0) if batch_list[0].t is not None else None
        xs = torch.concatenate([b.x for b in batch_list], dim=0) if batch_list[0].x is not None else None
        ys = torch.concatenate([b.y for b in batch_list], dim=0) if batch_list[0].y is not None else None
        us = torch.concatenate([b.u for b in batch_list], dim=0) if batch_list[0].u is not None else None
        ps = torch.concatenate([b.p for b in batch_list], dim=0) if batch_list[0].p is not None else None
        return DynData(t=ts, x=xs, y=ys, u=us, p=ps, meta=ms)

    def get_step(self, start: int, end: Optional[int] = None) -> "DynData":
        if end is None:
            end = start + 1
        tmp = DynData(
            t = self.t[:, start:end] if self.t is not None else None,
            x = self.x[:, start:end] if self.x is not None else None,
            y = self.y[:, start:end] if self.y is not None else None,
            u = self.u[:, start:end] if self.u is not None else None,
            p = self.p,
            ei = _slice(self.ei, start, end) if self._has_graph else None,
            ew = _slice(self.ew, start, end) if self.ew is not None else None,
            ea = _slice(self.ea, start, end) if self.ea is not None else None,
            n_nodes = self.n_nodes,
            meta = self.meta
        )
        if end == start + 1:
            # Squeeze time dimension if only one step is selected
            tmp.squeeze_time()
        return tmp

    def squeeze_time(self) -> "DynData":
        self.t = self.t.squeeze(1) if self.t is not None else None
        self.x = self.x.squeeze(1) if self.x is not None else None
        self.y = self.y.squeeze(1) if self.y is not None else None
        self.u = self.u.squeeze(1) if self.u is not None else None
        if self._has_graph:
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1) if self.x is not None else None
            self.y_reshape = self.y.shape[:-1] + (self.n_nodes, -1) if self.y is not None else None
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1) if self.u is not None else None
            self.p_reshape = self.p.shape[:-1] + (self.n_nodes, -1) if self.p is not None else None
        return self

    def truncate(self, num_step):
        return self.get_step(0, num_step)

    def unfold(self, window: int, stride: int) -> "DynData":
        """
        Unfold the data into overlapping windows.

        The regular data array is assumed to be of shape (batch_size, n_steps, ...).
        unfold produces a tensor of shape (batch_size, n_window, ..., window).

            - In normal mode, merge the first two dimensions and permute the last two gives
              (batch_size*n_window, window, ...)
            - In graph mode, we always have batch_size=1, and we permute to
              (1, window, n_window x ...), where x is the merged edge dimension over all windows.

        Accompanying the unfold in graph mode, the edge indices and attributes are also unfolded.
        The result is a graph of n_window times the size of the original graph.

        Args:
            window (int): Size of the sliding window.
            stride (int): Step size for the sliding window.

        Returns:
            DynData: A new DynData instance with unfolded data.
        """
        if self._has_graph:
            # Graph mode:
            _unf = lambda z: z.unfold(1, window, stride).permute(0, 3, 1, 2).reshape(1, window, -1) if z is not None else None
            t_unfolded = self.t.unfold(1, window, stride).reshape(-1, window).transpose(1, 0).unsqueeze(0) if self.t is not None else None
            x_unfolded = _unf(self.x)
            y_unfolded = _unf(self.y)
            u_unfolded = _unf(self.u)
            n_windows  = x_unfolded.size(-1) // self.x.size(-1)
            p_unfolded = self.p.tile((n_windows,)) if self.p is not None else None

            ei_unfolded = _unfold(self.ei, window, stride, offset=self.n_nodes)
            ew_unfolded = _unfold(self.ew, window, stride)
            ea_unfolded = _unfold(self.ea, window, stride)
            return DynData(
                t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded,
                ei=ei_unfolded, ew=ew_unfolded, ea=ea_unfolded, meta=self.meta)

        # Normal mode:
        _unf = lambda z: z.unfold(1, window, stride).reshape(-1, z.size(-1), window).permute(0, 2, 1) if z is not None else None
        t_unfolded = self.t.unfold(1, window, stride).reshape(-1, window) if self.t is not None else None
        x_unfolded = _unf(self.x)
        y_unfolded = _unf(self.y)
        u_unfolded = _unf(self.u)
        n_windows  = x_unfolded.size(0) // self.x.size(0)
        p_unfolded = self.p.repeat_interleave(n_windows, dim=0) if self.p is not None else None
        return DynData(t=t_unfolded, x=x_unfolded, y=y_unfolded, u=u_unfolded, p=p_unfolded, meta=self.meta)

    def set_x(self, value: torch.Tensor) -> "DynData":
        self.x = value
        if self._has_graph:
            self.x_reshape = self.x.shape[:-1] + (self.n_nodes, -1)
        return self

    def set_u(self, value: Optional[torch.Tensor] = None) -> "DynData":
        if value is None:
            return self
        self.u = value
        if self._has_graph:
            self.u_reshape = self.u.shape[:-1] + (self.n_nodes, -1)
        return self

    @property
    def xg(self) -> torch.Tensor:
        """
        Get the feature tensor with shape (batch_size, n_steps, n_nodes, n_features_per_node).
        """
        return self.x.reshape(*self.x_reshape)

    @property
    def yg(self) -> torch.Tensor:
        """
        Get the auxiliary tensor with shape (batch_size, n_steps, n_nodes, n_aux_per_node).
        """
        return self.y.reshape(*self.y_reshape)

    @property
    def ug(self) -> Union[torch.Tensor, None]:
        """
        Get the control tensor with shape (batch_size, n_steps, n_nodes, n_controls).
        """
        return self.u.reshape(*self.u_reshape)

    @property
    def pg(self) -> Union[torch.Tensor, None]:
        """
        Get the parameter tensor with shape (batch_size, n_nodes, n_parameters).
        """
        return self.p.reshape(*self.p_reshape)

    def g(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reshape a tensor to have shape (..., n_nodes, -1).

        Args:
            z (torch.Tensor): Input tensor of shape (..., n_nodes * features).

        Returns:
            torch.Tensor: Reshaped tensor of shape (..., n_nodes, features).
        """
        out_shape = z.shape[:-1] + (self.n_nodes, -1)
        return z.reshape(*out_shape)

    def G(self, z: torch.Tensor) -> torch.Tensor:
        """
        The reverse of g()
        """
        out_shape = z.shape[:-2] + (-1,)
        return z.reshape(*out_shape)
