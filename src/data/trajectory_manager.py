import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data as PyGData
from typing import Optional, Callable, Union, Tuple

# Import the Scaler and DelayEmbedder from preprocessing.py
from .preprocessing import Scaler, DelayEmbedder
from src.models.weak import generate_weak_weights


class TrajectoryManager:
    """
    A class to manage trajectory data loading, preprocessing, and
    dataloader creation.
    
    The workflow includes:
      - Loading raw data from a binary file.
      - Preprocessing (trimming trajectories, subsetting, etc.).
      - Scaling using the provided Scaler class.
      - Delay embedding using the provided DelayEmbedder class.
      - (Optionally) generating weak-form parameters.
      - Creating a dataset and splitting into train/validation/test sets.
      - Creating dataloaders tailored for NN, LSTM, or GNN models.
      
    The class is configured via a YAML configuration file.
    """

    def __init__(self, config_path: str, device: torch.device = torch.device("cpu")):
        """
        Initialize the TrajectoryManager by loading the YAML config.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            device (torch.device): Torch device to use.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.device = device

        # Placeholders for processing objects and data.
        self.scaler: Optional[Scaler] = None
        self.delay_embedder: Optional[DelayEmbedder] = None
        self.weak_dyn_param: Optional[dict] = None

        self.x: Optional[list[np.ndarray]] = None
        self.u: Optional[list[np.ndarray]] = None
        self.t: Optional[np.ndarray] = None
        self.off: Optional[np.ndarray] = None
        self.scl: Optional[np.ndarray] = None

        self.metadata: dict = {}
        self.dataset: Optional[torch.Tensor] = None

    def load_data(self, path: str) -> None:
        """
        Load raw data from a binary file.
        
        The file is assumed to store (in order):
            x: array-like or list of array-like, shape (n_samples, n_input_features)
            Training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

            t: float, numpy array of shape (n_samples,), or list of numpy arrays
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.

            u: array-like or list of array-like, shape (n_samples, n_control_features), optional (default None)
            Control variables/inputs. 
            If training data contains multiple trajectories (i.e. if x is a list of
            array-like), then u should be a list containing control variable data
            for each trajectory. Individual trajectories may contain different
            numbers of samples.
        
        Args:
            path (str): Path to the data file.
        """
        # Load the binary data from the file.
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        # Extract x, t, and u from the loaded data.
        self.x = data['x']
        self.t = data['t']
        self.u = data.get('u', None)

        # Ensure x and t are lists if they are not already.
        if not isinstance(self.x, list):
            self.x = [self.x]
        if not isinstance(self.t, list):
            self.t = [self.t]

        # If u is not provided, create a list of None values with the same length as x.
        if self.u is None:
            self.u = [None] * len(self.x)
        elif not isinstance(self.u, list):
            self.u = [self.u]

        # Ensure that the lengths of x, t, and u match.
        if len(self.x) != len(self.t) or len(self.x) != len(self.u):
            raise ValueError("The lengths of x, t, and u must match.")

        # Convert t to numpy arrays if they are not already.
        self.t = [np.array(ti) if not isinstance(ti, np.ndarray) else ti for ti in self.t]

        # Ensure that the time values in t are strictly increasing.
        for ti in self.t:
            if not np.all(np.diff(ti) > 0):
                raise ValueError("Time values in t must be strictly increasing.")

    def data_truncation(self) -> None:
        """
        Truncate the loaded data according to the configuration.
        
        This includes:
          - Subsetting the number of trajectories and horizon (n_steps).
          - Populating basic metadata (dt, tf, shapes, etc.).
        """
        cfg = self.config.get("data", {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)
        delay: int = cfg.get("delay", 0)

        if self.x is None or self.u is None or self.t is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if n_samples is not None:
            if n_steps is not None:
                self.x = [traj[:n_steps] for traj in self.x[:n_samples]]
                self.u = [inp[:n_steps] for inp in self.u[:n_samples]]
                self.t = self.t[:n_steps]
            else:
                self.x = self.x[:n_samples]
                self.u = self.u[:n_samples]

        # Update metadata.
        self.metadata["t"] = torch.tensor(self.t, dtype=torch.double, device=self.device)
        self.metadata["n_samples"] = len(self.x)
        self.metadata["n_steps"] = [len(traj) for traj in self.x]
        self.metadata["n_input_features"] = int(self.x[0].shape[-1])
        self.metadata["n_control_features"] = int(self.u[0].shape[-1]) if self.u[0].ndim > 1 else 0
        self.metadata["delay"] = delay
        self.metadata["off"] = self.off
        self.metadata["scl"] = self.scl

    def apply_scaling(self) -> None:
        """
        Apply scaling to the solutions and inputs using the Scaler class.
        
        The scaling mode is read from the configuration.
        """
        scale_cfg = self.config.get("scaling", {})
        mode: str = scale_cfg.get("mode", "01")

        # Create and fit a Scaler instance.
        self.scaler = Scaler(mode=mode)
        self.scaler.fit(self.x)

        # Transform the trajectories.
        self.x = self.scaler.transform(self.x)

    def apply_delay_embedding(self) -> None:
        """
        Apply delay embedding to the trajectories if a positive delay is specified.
        
        The solutions are transformed using the DelayEmbedder, and the inputs are trimmed
        accordingly.
        """
        delay = self.metadata.get("delay", 0)
        if delay > 0:
            # Create a DelayEmbedder instance.
            self.delay_embedder = DelayEmbedder(delay=delay)
            # Delay-embed the solutions.
            self.x = self.delay_embedder.transform(self.x)
            # For inputs, we simply remove the first `delay` time steps.
            self.u = [inp[delay:] for inp in self.u]
        # If delay==0, nothing is changed.

    def generate_weak_form_params(self) -> None:
        """
        If weak form is enabled in the configuration, generate and store the weak form parameters.
        
        This method uses the generate_weak_weights function from the weak module.
        """
        weak_cfg = self.config.get("weak_form", {})
        if not weak_cfg.get("enabled", False):
            return

        weakParam = weak_cfg.get("parameters", None)
        if weakParam is None:
            raise ValueError("Weak form is enabled but parameters are not provided.")

        # Extract parameters from the configuration.
        # If the parameter list is length 4, set alpha=1 by default.
        if len(weakParam) == 4:
            N, dN, ordpol, ordint = weakParam
            alpha = 1  # alpha is not used by generate_weak_weights, but we store it for metadata.
        elif len(weakParam) == 5:
            N, dN, ordpol, ordint, alpha = weakParam
        else:
            raise ValueError("weakParam must be of length 4 or 5.")

        # Call the generate_weak_weights function to get C, D, and K.
        C, D, K = generate_weak_weights(
            tf=self.metadata["tf"],  # TODO: non-uniform tf?
            n_steps=self.metadata["n_steps"],
            n_integration_points=N,
            integration_stride=dN,
            poly_order=ordpol,
            int_rule_order=ordint,
            delay=self.metadata["delay"]
        )

        # Convert weights to torch tensors and store in the weak_dyn_param dictionary.
        self.weak_dyn_param = {
            "C": torch.tensor(C, dtype=torch.double, device=self.device),
            "D": torch.tensor(D, dtype=torch.double, device=self.device),
            "K": K,
            "N": N,
            "dN": dN,
            "ordPoly": ordpol,
            "ordInt": ordint,
            "alpha": alpha,
        }
        self.metadata["weakDynParam"] = self.weak_dyn_param

    def create_dataset(self) -> torch.Tensor: # TODO: this needs to be looked at again, as only some of the projects use this data structure
        """
        Create a dataset by concatenating the trajectories and control inputs.
               
        Returns:
            A torch.Tensor representing the dataset.
        """
        
        if self.metadata["n_control_features"] == 0: # autonomous
            dataset = self.x
        else:
            # Concatenate along the feature dimension.
            dataset = [np.concatenate([traj, inp], axis=-1) for traj, inp in zip(self.x, self.u)]
        # Convert to torch.Tensor.
        self.dataset = torch.tensor(dataset, dtype=torch.double, device=self.device)
        self.metadata["n_features"] = self.dataset.shape[-1]
        return self.dataset

    def split_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the dataset into training, validation, and test sets.
        
        The training fraction is specified in the YAML config (default 0.75).
        
        Returns:
            A tuple of (train_set, valid_set, test_set).
        """
        split_cfg = self.config.get("split", {})
        train_frac: float = split_cfg.get("train_frac", 0.75)
        n_data = self.dataset.shape[0]
        n_train = int(n_data * train_frac)
        remaining = n_data - n_train
        n_val = remaining // 2
        n_test = remaining - n_val

        self.metadata["n_train"] = n_train
        self.metadata["n_val"] = n_val
        self.metadata["n_test"] = n_test

        train_set = self.dataset[:n_train]
        valid_set = self.dataset[n_train:n_train+n_val]
        test_set = self.dataset[n_train+n_val:]
        return train_set, valid_set, test_set

    def _create_graph(self,
        features: torch.Tensor,
        adj: Optional[Union[torch.Tensor, np.ndarray]] = None,
        custom_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], PyGData]] = None
    ) -> PyGData:
        """
        Create a graph from a trajectory using an optional adjacency matrix.
        
        This function converts an adjacency matrix (if provided) into a PyTorch Geometric 
        edge_index and edge_attr. It also allows a custom graph construction function to override
        the default behavior.
        
        Args:
            features (torch.Tensor): Node feature matrix of shape (n_nodes, n_features) where n_nodes is the number of nodes.
            adj (torch.Tensor or np.ndarray, optional): A square adjacency matrix of shape (n_nodes, n_nodes). 
                Nonzero entries indicate edges. Their values are used as edge attributes.
            custom_fn (callable, optional): A custom function that takes (traj, adj) and returns a PyGData object.
                If provided, this function overrides the default graph construction.
        
        Returns:
            PyGData: A PyTorch Geometric Data object containing:
                - x: node n_features (same as traj)
                - edge_index: tensor of shape (2, n_edges) listing edge connections
                - edge_attr: tensor of shape (n_edges, 1) containing edge attributes (if adj is provided)
        """
        # Use custom function if provided.
        if custom_fn is not None:
            return custom_fn(features, adj)
        
        # If an adjacency matrix is provided, use it to build edge_index and edge_attr.
        if adj is not None:
            # If provided as a numpy array, convert it to a torch tensor.
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.double, device=features.device)
            else:
                # Ensure the adjacency matrix is on the same device as traj.
                adj = adj.to(features.device)
            
            # Verify that the adjacency matrix is square and matches the number of nodes.
            n_nodes = features.size(0)
            if adj.size(0) != n_nodes or adj.size(1) != n_nodes:
                raise ValueError(f"Adjacency matrix shape {adj.shape} does not match the number of nodes {n_nodes}.")

            # Find nonzero entries to build edge_index.
            edge_index = (adj != 0).nonzero(as_tuple=False).t().contiguous()  # shape: (2, n_edges)
            # Use the nonzero values as edge attributes.
            nonzero_vals = adj[adj != 0].view(-1, 1)  # shape: (n_edges, 1)
            return PyGData(x=features, edge_index=edge_index, edge_attr=nonzero_vals)
        
        else:
            return PyGData(x=features)

    def create_dataloaders(
        self, train_set: torch.Tensor, valid_set: torch.Tensor, test_set: torch.Tensor
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for the dataset based on the model type.
        
        The model type is specified in the YAML config under "dataloader/model_type"
        and can be one of "NN", "LSTM", or "GNN".
        
        Returns:
            A tuple (train_loader, valid_loader, test_loader).
        """
        dl_cfg = self.config.get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 32)
        model_type: str = dl_cfg.get("model_type", "NN").upper()

        if model_type == "NN":
            from torch.utils.data import TensorDataset
            train_loader = DataLoader(TensorDataset(train_set), batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(TensorDataset(valid_set), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(test_set), batch_size=batch_size, shuffle=False)

        elif model_type == "LSTM":
            from torch.utils.data import TensorDataset
            n_steps: int = dl_cfg.get("n_steps", 1)
            X_train, y_train = self._create_sequences(train_set, n_steps)
            X_valid, y_valid = self._create_sequences(valid_set, n_steps)
            X_test, y_test = self._create_sequences(test_set, n_steps)

            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        elif model_type == "GNN":
            # For GNN, each trajectory is converted into a PyTorch Geometric Data object.
            from torch_geometric.loader import DataLoader as GeoDataLoader
          
            # Allow optional GNN configuration to specify an adjacency matrix or custom function.
            gnn_cfg = dl_cfg.get("gnn", {})
            # For example, one might provide an "adjacency" key if a constant adj matrix is desired;
            # otherwise, None will cause the default chain-graph to be built.
            adj = gnn_cfg.get("adjacency", None)
            # A custom graph construction function can be provided via configuration if desired.
            custom_fn = gnn_cfg.get("custom_fn", None)

            # Build graphs from each trajectory sample.
            train_data = [self._create_graph(sample, adj=adj, custom_fn=custom_fn) for sample in train_set]
            valid_data = [self._create_graph(sample, adj=adj, custom_fn=custom_fn) for sample in valid_set]
            test_data = [self._create_graph(sample, adj=adj, custom_fn=custom_fn) for sample in test_set]

            train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
            valid_loader = GeoDataLoader(valid_data, batch_size=batch_size, shuffle=False)
            test_loader = GeoDataLoader(test_data, batch_size=batch_size, shuffle=False)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return train_loader, valid_loader, test_loader


    def _create_sequences(
        self, data: torch.Tensor, n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding window sequences for LSTM models.
        
        For each trajectory (assumed to be of shape (T, n_features)), this
        method creates sequences of length `n_steps` (X) with the subsequent
        time step as the target (y).
        
        Args:
            data (torch.Tensor): Tensor of shape (n_samples, T, n_features).
            n_steps (int): Length of the input sequence.
        
        Returns:
            A tuple (X, y) where:
              - X is of shape (n_samples, n_steps, n_features)
              - y is of shape (n_samples, n_features)
        """
        X_list = []
        y_list = []
        # Loop over each trajectory.
        for traj in data:
            T = traj.shape[0]
            if T < n_steps + 1:
                continue
            for i in range(T - n_steps):
                X_list.append(traj[i:i + n_steps])
                y_list.append(traj[i + n_steps])
        X_tensor = torch.stack(X_list)
        y_tensor = torch.stack(y_list)
        return X_tensor, y_tensor

    def process_all(self, path: str) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Run the full processing pipeline:
          1. Load raw data.
          2. Preprocess and subset.
          3. Apply scaling.
          4. Apply delay embedding.
          5. (Optionally) generate weak-form parameters.
          6. Create a dataset.
          7. Split the dataset.
          8. Create dataloaders.
        
        Args:
            path (str): Path to the raw data file.
        
        Returns:
            A tuple containing:
              - A tuple of (train_loader, valid_loader, test_loader).
              - A tuple of (train_set, valid_set, test_set) tensors.
              - A metadata dictionary.
        """
        self.load_data(path)
        self.data_truncation()
        self.apply_scaling()
        self.apply_delay_embedding()
        self.generate_weak_form_params()
        self.create_dataset()
        train_set, valid_set, test_set = self.split_dataset()
        loaders = self.create_dataloaders(train_set, valid_set, test_set)
        return loaders, (train_set, valid_set, test_set), self.metadata
