import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data as PyGData
from typing import Optional, Callable, Union, Tuple

# Import the Scaler and DelayEmbedder from preprocessing.py
from .preprocessing import Scaler, DelayEmbedder

# Import the generate_weak_weights function from your weak module.
# Adjust the import path as needed (e.g., if your package structure is different).
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

        self.solutions: Optional[np.ndarray] = None
        self.inputs: Optional[np.ndarray] = None
        self.time_vector: Optional[np.ndarray] = None
        self.off: Optional[np.ndarray] = None
        self.scl: Optional[np.ndarray] = None

        self.metadata: dict = {}
        self.dataset: Optional[torch.Tensor] = None

    def load_data(self, path: str) -> None:
        """
        Load raw data from a binary file.
        
        The file is assumed to store (in order):
            - solutions: numpy.ndarray of shape (num_sequences, seq_length, features)
            - inputs: numpy.ndarray of shape (num_sequences, seq_length, inputs)
            - time_vector: numpy.ndarray of shape (seq_length,)
            - scaling parameters: (off, scl)
        
        Args:
            path (str): Path to the data file.
        """
        with open(path, "rb") as f:
            self.solutions = np.load(f)
            self.inputs = np.load(f)
            self.time_vector = np.load(f)
            off_scl = np.load(f, allow_pickle=True)
            self.off, self.scl = off_scl

    def preprocess_data(self) -> None:
        """
        Preprocess the loaded data according to the configuration.
        
        This includes:
          - Optionally removing the first time step.
          - Subsetting the number of trajectories and timesteps.
          - Populating basic metadata (dt, tf, shapes, etc.).
        """
        cfg = self.config.get("data", {})
        rm1st: bool = cfg.get("rm1st", True)
        nTraj: Optional[int] = cfg.get("nTraj", None)
        nSteps: Optional[int] = cfg.get("nSteps", None)
        delay: int = cfg.get("delay", 0)

        if self.solutions is None or self.inputs is None or self.time_vector is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if rm1st:
            self.solutions = self.solutions[:, 1:]
            self.inputs = self.inputs[:, 1:]
            self.time_vector = self.time_vector[:-1]

        if nTraj is not None:
            if nSteps is not None:
                self.solutions = self.solutions[:nTraj, :nSteps]
                self.inputs = self.inputs[:nTraj, :nSteps]
                self.time_vector = self.time_vector[:nSteps]
            else:
                self.solutions = self.solutions[:nTraj]
                self.inputs = self.inputs[:nTraj]

        # Update metadata.
        self.metadata["dt"] = float(self.time_vector[1] - self.time_vector[0])
        self.metadata["tf"] = float(self.time_vector[-1])
        self.metadata["nTraj"] = int(self.solutions.shape[0])
        self.metadata["nTimesteps"] = int(self.solutions.shape[1])
        self.metadata["nStates"] = int(self.solutions.shape[-1])
        self.metadata["nInputs"] = int(self.inputs.shape[-1]) if self.inputs.ndim > 1 else 0
        self.metadata["time_vector"] = torch.tensor(self.time_vector, dtype=torch.double, device=self.device)
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
        self.scaler.fit(self.solutions)

        # Transform the data.
        self.solutions = self.scaler.transform(self.solutions)
        self.inputs = self.scaler.transform(self.inputs)

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
            self.solutions = self.delay_embedder.transform(self.solutions)
            # For inputs, we simply remove the first `delay` time steps.
            self.inputs = self.inputs[:, delay:]
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
            tf=self.metadata["tf"],
            num_time_points=self.metadata["nTimesteps"],
            num_integration_points=N,
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

    def create_dataset(self) -> torch.Tensor:
        """
        Create a dataset by concatenating the solutions and inputs.
        
        If the "auto" flag is enabled in the configuration, only the solutions
        (states) are used.
        
        Returns:
            A torch.Tensor representing the dataset.
        """
        data_cfg = self.config.get("data", {})
        auto: bool = data_cfg.get("auto", False)
        if auto:
            dataset = self.solutions
            self.metadata["nInputs"] = 0
        else:
            # Concatenate along the feature dimension.
            dataset = np.concatenate([self.solutions, self.inputs], axis=-1)
        # Convert to torch.Tensor.
        self.dataset = torch.tensor(dataset, dtype=torch.double, device=self.device)
        self.metadata["nFeatures"] = self.dataset.shape[-1]
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

        self.metadata["nTrn"] = n_train
        self.metadata["nVal"] = n_val
        self.metadata["nTst"] = n_test

        train_set = self.dataset[:n_train]
        valid_set = self.dataset[n_train:n_train+n_val]
        test_set = self.dataset[n_train+n_val:]
        return train_set, valid_set, test_set

    def _create_graph(self,
        traj: torch.Tensor,
        adj: Optional[Union[torch.Tensor, np.ndarray]] = None,
        custom_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], PyGData]] = None
    ) -> PyGData:
        """
        Create a graph from a trajectory using an optional adjacency matrix.
        
        This function converts an adjacency matrix (if provided) into a PyTorch Geometric 
        edge_index and edge_attr. It also allows a custom graph construction function to override
        the default behavior.
        
        Args:
            traj (torch.Tensor): Node feature matrix of shape (T, num_features) where T is the number of nodes.
            adj (torch.Tensor or np.ndarray, optional): A square adjacency matrix of shape (T, T). 
                Nonzero entries indicate edges. Their values are used as edge attributes.
            custom_fn (callable, optional): A custom function that takes (traj, adj) and returns a PyGData object.
                If provided, this function overrides the default graph construction.
        
        Returns:
            PyGData: A PyTorch Geometric Data object containing:
                - x: node features (same as traj)
                - edge_index: tensor of shape (2, num_edges) listing edge connections
                - edge_attr: tensor of shape (num_edges, 1) containing edge attributes (if adj is provided)
        """
        # Use custom function if provided.
        if custom_fn is not None:
            return custom_fn(traj, adj)
        
        # If an adjacency matrix is provided, use it to build edge_index and edge_attr.
        if adj is not None:
            # If provided as a numpy array, convert it to a torch tensor.
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.double, device=traj.device)
            else:
                # Ensure the adjacency matrix is on the same device as traj.
                adj = adj.to(traj.device)
            
            # Verify that the adjacency matrix is square and matches the number of nodes.
            T = traj.size(0)
            if adj.size(0) != T or adj.size(1) != T:
                raise ValueError(f"Adjacency matrix shape {adj.shape} does not match the number of nodes {T}.")

            # Find nonzero entries to build edge_index.
            edge_index = (adj != 0).nonzero(as_tuple=False).t().contiguous()  # shape: (2, num_edges)
            # Use the nonzero values as edge attributes.
            nonzero_vals = adj[adj != 0].view(-1, 1)  # shape: (num_edges, 1)
            return PyGData(x=traj, edge_index=edge_index, edge_attr=nonzero_vals)
        
        # Fallback: Create a chain graph (default behavior) if no adjacency matrix is provided.
        T = traj.size(0)
        if T > 1:
            # Connect consecutive nodes.
            edge_index = torch.tensor(
                [list(range(T - 1)), list(range(1, T))],
                dtype=torch.long,
                device=traj.device,
            )
            # For undirected graphs, add reverse edges.
            edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=traj.device)
        return PyGData(x=traj, edge_index=edge_index)

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
            seq_length: int = dl_cfg.get("seq_length", 1)
            X_train, y_train = self._create_sequences(train_set, seq_length)
            X_valid, y_valid = self._create_sequences(valid_set, seq_length)
            X_test, y_test = self._create_sequences(test_set, seq_length)

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
        self, data: torch.Tensor, seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding window sequences for LSTM models.
        
        For each trajectory (assumed to be of shape (T, features)), this
        method creates sequences of length `seq_length` (X) with the subsequent
        time step as the target (y).
        
        Args:
            data (torch.Tensor): Tensor of shape (nTraj, T, features).
            seq_length (int): Length of the input sequence.
        
        Returns:
            A tuple (X, y) where:
              - X is of shape (num_sequences, seq_length, features)
              - y is of shape (num_sequences, features)
        """
        X_list = []
        y_list = []
        # Loop over each trajectory.
        for traj in data:
            T = traj.shape[0]
            if T < seq_length + 1:
                continue
            for i in range(T - seq_length):
                X_list.append(traj[i:i + seq_length])
                y_list.append(traj[i + seq_length])
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
        self.preprocess_data()
        self.apply_scaling()
        self.apply_delay_embedding()
        self.generate_weak_form_params()
        self.create_dataset()
        train_set, valid_set, test_set = self.split_dataset()
        loaders = self.create_dataloaders(train_set, valid_set, test_set)
        return loaders, (train_set, valid_set, test_set), self.metadata
