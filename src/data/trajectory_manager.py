import os, logging, torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as GeoDataLoader
from typing import Optional, Callable, Union, Tuple, Dict, List

# Import the Scaler and DelayEmbedder from preprocessing.py
from .preprocessing import Scaler, DelayEmbedder
from ..utils.weak import generate_weak_weights

logging = logging.getLogger(__name__)
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

    def __init__(self, metadata: Dict, device: torch.device = torch.device("cpu"), adj: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """
        Initialize the TrajectoryManager by loading the YAML config.
        
        Args:
            metadata (dict): Configuration dictionary.
            device (torch.device): Torch device to use.
            adj (torch.Tensor or np.ndarray, optional): Adjacency matrix for GNN models.
                If not provided, will try to get from config.
        """
        self.metadata = metadata
        self.dtype = torch.double if self.metadata['config']['data'].get('double_precision', False) else torch.float
        if self.metadata['config']['data']['delay'] < 0:
            raise ValueError("Delay must be non-negative.")
        self.delay = self.metadata['config']['data']['delay']        
        self.model_type = self.metadata['config']['model']['type']
        self.enable_weak_form = self.metadata['config']['weak_form']['enabled']
        self.device = device
        self.data_path = self.metadata['config']['data']['path']
        self.adj = adj  # Store the adjacency matrix
        self.n_nodes = self.metadata['config']['data'].get('n_nodes', None)
        
    def load_data(self, path: str) -> None:
        """
        Load raw data from a binary file.
        
        The file is assumed to store (in order):
            x: array-like or list of array-like, shape (n_samples, n_state_features)
            training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

            t: float, numpy array of shape (n_samples,), or list of numpy arrays
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time (seconds in physical time) at which each sample was
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
        data = np.load(path, allow_pickle=True)

        # Extract x, t, and u from the loaded data.
        self.x = data['x']
        self.t = data['t']
        self.u = data.get('u', None)
        logging.info("Raw data loaded.")

        # Process x
        if isinstance(self.x, np.ndarray):
            if self.x.ndim == 3:  # multiple trajectories provided as a 3D array.
                self.x = [_x for _x in self.x]
            else:  # single trajectory provided as a 2D array.
                self.x = [self.x]
        elif not isinstance(self.x, list):
            self.x = [self.x]

        # Process u
        if self.u is None:
            self.u = [None] * len(self.x)
        elif isinstance(self.u, np.ndarray):
            if self.u.ndim == 3:  # multiple trajectories provided as a 3D array.
                self.u = [_u for _u in self.u]
            else:  # single trajectory provided as a 2D array.
                self.u = [self.u]
        elif not isinstance(self.u, list):
            self.u = [self.u]

        # Process t
        if isinstance(self.t, (float, int)):
            # Case 1: t is a float/int specifying timestep between samples
            self.dt = float(self.t)
            self.t = [np.arange(traj.shape[0]) * self.dt for traj in self.x]
            self.dt = [self.dt for _ in self.t]
        elif isinstance(self.t, list):
            # Case 2: t is a list of arrays for multiple trajectories
            self.t = [np.array(ti) for ti in self.t]  # Ensure each element is a numpy array
            self.dt = [ti[1] - ti[0] for ti in self.t]
        elif isinstance(self.t, np.ndarray):
            if self.t.ndim == 1:
                # Case 3: t is a 1D array of shape (n_samples,)
                self.dt = [self.t[1] - self.t[0]]
                self.t = [self.t]
            elif self.t.ndim == 2:
                # Case 4: t is a 2D array of shape (n_trajectories, n_samples)
                self.t = [ti for ti in self.t]  # Split into list of 1D arrays
                self.dt = [ti[1] - ti[0] for ti in self.t]
            else:
                raise ValueError(f"Unsupported array dimension for t: {self.t.ndim}")
        else:
            raise TypeError("t must be a float, numpy array, or list of numpy arrays")

    def data_truncation(self) -> None:
        """
        Truncate the loaded data according to the configuration.
        
        This includes:
          - Subsetting the number of trajectories and horizon (n_steps).
          - Populating basic metadata (dt, tf, shapes, etc.).
        """
        cfg = self.metadata['config'].get("data", {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)

        if self.x is None or self.u is None or self.t is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            self.x = self.x[:n_samples]
            self.u = self.u[:n_samples]
            self.t = self.t[:n_samples]
            self.dt = self.dt[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.x = [traj[:n_steps] for traj in self.x]
            # Apply truncation only if control inputs exist.
            self.u = [inp[:n_steps] if inp is not None else None for inp in self.u]
            self.t = [ti[:n_steps] for ti in self.t]
        
        # Check if uniform t and dt, if so, only save one
        self.dt = [self.dt[0]] if len(set(self.dt)) == 1 else self.dt
        self.t = [self.t[0]] if len(set(map(len, self.t))) == 1 else self.t

        self.metadata['delay'] = self.delay
        self.metadata["n_samples"] = len(self.x)
        self.metadata["dt_and_n_steps"] = [[dt, len(t)] for dt, t in zip(self.dt, self.t)]
        self.metadata["n_state_features"] = int(self.x[0].shape[-1])
        self.metadata["n_control_features"] = int(self.u[0].shape[-1]) if self.u[0].ndim > 1 else 0

        logging.info(f"Number of samples: {self.metadata['n_samples']}")
        logging.info(f"Number of state features: {self.metadata['n_state_features']}")
        logging.info(f"Number of control features: {self.metadata['n_control_features']}")
        logging.info(f"Delay embedding size: {self.metadata['delay']}")

    def apply_scaling(self) -> None:
        """
        Apply scaling to the solutions and inputs using the Scaler class.

        The scaling mode is read from the configuration. If a checkpoint is provided via 
        "scaling: load_from_checkpoint" in the config, scaling parameters are loaded from that .pt file.
        Otherwise, the scaler is fitted using the data.
        """

        if "scaler" in self.metadata:
            logging.info("Loading scaling parameters from checkpoint")
            self.scaler = Scaler(
            mode=self.metadata["scaler"]['mode'],
            scl=self.metadata["scaler"]['scale'],
            off=self.metadata["scaler"]['offset']
            )
        else:
            scaling_mode = self.metadata['config']['scaling']['mode']
            logging.info(f"Applying scaling with mode: {scaling_mode}.")
            self.scaler = Scaler(mode=scaling_mode)
            self.scaler.fit(self.x)
            self.metadata["scaler"] = {
                "offset": self.scaler._off,
                "scale": self.scaler._scl,
                "mode": scaling_mode
            }

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
            logging.info(f"Applying delay embedding with delay={delay}.")
            # Create a DelayEmbedder instance.
            self.delay_embedder = DelayEmbedder(delay=delay)
            # Delay-embed the solutions.
            self.x = self.delay_embedder.transform(self.x)
            # For inputs, we simply remove the first `delay` time steps.
            self.u = [inp[delay:] for inp in self.u]
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-delay] for ti in self.t]
            # Update the metadata.
            self.metadata["delay"] = delay
            self.metadata["dt_and_n_steps"] = [[dt, len(t)] for dt, t in zip(self.dt, self.t)]
        # If delay==0, nothing is changed.

    def generate_weak_form_params(self) -> None:
        """
        If weak form is enabled in the configuration, generate and store the weak form parameters.
        
        This method uses the generate_weak_weights function from the weak module.
        """
        logging.info("Generating weak form parameters.")
        weak_cfg = self.metadata['config'].get("weak_form", {})
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

        if len(self.metadata["dt_and_n_steps"]) > 1: # TODO
            raise ValueError("Weak form generation is not currently supported for trajectories with different lengths.")
        
        # Call the generate_weak_weights function to get C, D, and K.
        C, D, K = generate_weak_weights(
            dt=self.metadata["dt_and_n_steps"][0][0],  # TODO: non-uniform dt?
            n_steps=self.metadata["dt_and_n_steps"][0][1], # TODO: non-uniform n_steps?
            n_integration_points=N,
            integration_stride=dN,
            poly_order=ordpol,
            int_rule_order=ordint,
        )

        # Convert weights to torch tensors and store in the weak_dyn_param dictionary.
        self.weak_dyn_param = {
            "C": torch.tensor(C, dtype=self.dtype, device=self.device),
            "D": torch.tensor(D, dtype=self.dtype, device=self.device),
            "K": K,
            "N": N,
            "dN": dN,
            "ordPoly": ordpol,
            "ordInt": ordint,
            "alpha": alpha,
        }
        self.metadata["weakDynParam"] = self.weak_dyn_param
        logging.info("Weak form parameters generated.")

    def create_dataset(self) -> torch.Tensor: # TODO: this needs to be looked at again, as only some of the projects use this data structure
        """
        Create a dataset by concatenating the trajectories and control inputs.
               
        Returns:
            A torch.Tensor representing the dataset.
        """
        if self.metadata["n_control_features"] == 0: # autonomous
            self.dataset = self.x
        else:
            # Concatenate along the feature dimension.
            self.dataset = [np.concatenate([traj, inp], axis=-1) for traj, inp in zip(self.x, self.u)]
        # Convert to torch.Tensor.
        self.dataset = [torch.tensor(entry, dtype=self.dtype, device=self.device) for entry in self.dataset]
        self.metadata["n_total_features"] = self.dataset[0].shape[-1] # This is observables and control inputs combined
        
    def split_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the dataset into training, validation, and test sets.
        
        The training fraction is specified in the YAML config (default 0.75).
        The split is performed by shuffling whole trajectories.

        """
        split_cfg = self.metadata['config'].get("split", {})
        train_frac: float = split_cfg.get("train_frac", 0.75)
        # Shuffle the trajectories randomly.
        perm = torch.randperm(self.metadata["n_samples"])
        dataset_rand = [self.dataset[i] for i in perm.tolist()]

        n_train = int(self.metadata["n_samples"] * train_frac)
        remaining = self.metadata["n_samples"] - n_train
        n_val = remaining // 2
        n_test = remaining - n_val

        self.metadata["n_train"] = n_train
        self.metadata["n_val"] = n_val
        self.metadata["n_test"] = n_test

        self.train_set = dataset_rand[:n_train]
        self.valid_set = dataset_rand[n_train:n_train+n_val]
        self.test_set = dataset_rand[n_train+n_val:]
        
    def create_dataloaders(self) -> None:
        """
        Create dataloaders for train, validation, and test sets based on the model type.
        
        The model type is specified in the YAML config under "dataloader/model_type"
        and can be one of "NN", "LSTM", or "GNN".
        
        This method creates and stores three dataloaders as class attributes:
        - self.train_loader
        - self.valid_loader
        - self.test_loader
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)

        if self.model_type == "NN":
            logging.info(f"Creating dataloaders for NN model with batch size {batch_size}.")
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
            self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

        elif self.model_type == "LSTM":
            logging.info(f"Creating dataloaders for LSTM model with batch size {batch_size}.")
            # Create sequences for each set
            self.train_set = TensorDataset(self._create_lstm_sequences(self.train_set))
            self.valid_set = TensorDataset(self._create_lstm_sequences(self.valid_set))
            self.test_set = TensorDataset(self._create_lstm_sequences(self.test_set))
            
            # Create dataloaders
            self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
            self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

        elif self.model_type == "GNN":
            logging.info(f"Creating dataloaders for GNN model with batch size {batch_size}.")
            gnn_cfg = dl_cfg.get("gnn", {})
            # Use provided adj matrix if available, otherwise try to get from config (TODO: does not support dynamic graphs yet)
            adj = self.adj if self.adj is not None else gnn_cfg.get("adjacency", None)
            
            # Build graph objects for each set
            self.train_set = self._create_gnn_sequences(self.train_set, adj)
            self.valid_set = self._create_gnn_sequences(self.valid_set, adj)
            self.test_set = self._create_gnn_sequences(self.test_set, adj)
            
            # Create dataloaders
            # Note: we use a GeoDataLoader for each trajectory in the dataset, with the batch size being the length of the trajectory.
            # This is because we want to ensure that each trajectory is processed as a whole, and not split into smaller batches.
            self.train_loader = [GeoDataLoader(_traj, batch_size=len(_traj), shuffle=False) for _traj in self.train_set]
            self.valid_loader = [GeoDataLoader(_traj, batch_size=len(_traj), shuffle=False) for _traj in self.valid_set]
            self.test_loader = [GeoDataLoader(_traj, batch_size=len(_traj), shuffle=False) for _traj in self.test_set]

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _create_gnn_sequences(
        self, dataset: list[torch.Tensor], adj: Optional[Union[torch.Tensor, np.ndarray]] = None) -> list[PyGData]:
        """
        Create a list of PyTorch Geometric Data objects from a list of tensors.
        
        Args:
            dataset (list[torch.Tensor]): List of tensors, each of shape (T, n_state_features + n_control_features)
            adj (torch.Tensor or np.ndarray, optional): Adjacency matrix of shape (n_nodes, n_nodes)
        
        Returns:
            list[PyGData]: List of PyTorch Geometric Data objects
        """
        seq_length = self.metadata['delay'] + 1
        n_nodes = self.n_nodes
        n_features_per_node = self.metadata['n_state_features'] // n_nodes
        data_list = []
        
        for traj in dataset:
            T = traj.shape[0]
            seq = []
            if T <= seq_length:
                continue  # Skip trajectories that are too short
            for i in range(T - seq_length):
                # Get states for the sequence window
                states = traj[i:i + seq_length, :self.metadata['n_state_features']]  # [seq_length, n_state_features]
                # Get control at the last step of the sequence
                controls = traj[i + seq_length - 1, -self.metadata['n_control_features']:]  # [n_control_features]
                
                # Reshape states to group features by node
                # From [seq_length, n_state_features] to [n_nodes, seq_length * n_features_per_node]
                states = states.reshape(seq_length, n_nodes, n_features_per_node)  # [seq_length, n_nodes, n_features_per_node]
                states = states.permute(1, 0, 2)  # [n_nodes, seq_length, n_features_per_node]
                states = states.reshape(n_nodes, -1)  # [n_nodes, seq_length * n_features_per_node]
                seq.append(self._create_pyg_data(states, controls, adj))
            data_list.append(seq)
        return data_list

    def _create_pyg_data(self,
        states: torch.Tensor,
        controls: torch.Tensor,
        adj: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> PyGData: # TODO: this implmentation does not support dynamic graphs yet.
        """
        Create a graph from a trajectory using an optional adjacency matrix.
        
        Args:
            states (torch.Tensor): Node feature matrix of shape (n_nodes, n_total_features)
            controls (torch.Tensor): Control input array of shape (n_control_features, )
            adj (torch.Tensor or np.ndarray, optional): Adjacency matrix of shape (n_nodes, n_nodes)
        
        Returns:
            PyGData: PyTorch Geometric Data object with node features and optional edge information
        """
        
        # Convert numpy array to torch tensor if needed
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adj, dtype=self.dtype, device=states.device)
        else:
            adj = adj.to(states.device)
        
        # Verify node count matches
        if adj.size(0) != self.n_nodes or adj.size(1) != self.n_nodes:
            raise ValueError(f"Adjacency matrix shape {adj.shape} does not match the number of nodes {self.n_nodes}.")
        
        # Convert adjacency matrix to edge_index and edge_attr using PyG
        from torch_geometric.utils import dense_to_sparse
        edge_index, edge_attr = dense_to_sparse(adj)
        return PyGData(x=states, u=controls, edge_index=edge_index, edge_attr=edge_attr)

    def _create_lstm_sequences(
        self, dataset: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create sliding window sequences for LSTM models.
        # For each trajectory (assumed to be of shape (T, n_state_features + n_control_features))
        # where T can vary, this method creates sequences of length `seq_length` (X) with the subsequent
        # time step as the target (y).
        #
        # Args:
        #   dataset (list[torch.Tensor]): List of tensors, each of shape (T, n_state_features + n_control_features)
        #   seq_length (int): Length of the input sequence.
        #
        # Returns:
        #   A tuple (X, y) where:
        #     - X is of shape (N, seq_length, n_state_features + n_control_features)
        #     - y is of shape (N, n_state_features)
        seq_length = self.metadata['delay'] + 1
        X_list = []
        y_list = []
        # Loop over each trajectory.
        for traj in dataset:
            T = traj.shape[0]
            if T < seq_length + 1:
                continue
            for i in range(T - seq_length):
                X_list.append(traj[i:i + seq_length])
                y_list.append(traj[i + seq_length, :self.metadata['n_state_features']])
        X_tensor = torch.stack(X_list)
        y_tensor = torch.stack(y_list)
        return X_tensor, y_tensor

    def process_all(self, steps: Optional[List[str]] = None) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Run the data processing pipeline with user-defined steps.
        
        Args:
            steps (List[str], optional): List of processing steps to execute in order.
                If None, uses default pipeline. Available steps:
                - 'load_data': Load raw data from file
                - 'data_truncation': Truncate data according to config
                - 'scaling': Apply scaling to data
                - 'delay_embedding': Apply delay embedding
                - 'weak_form': Generate weak form parameters
                - 'create_dataset': Create dataset from processed data
                - 'split_dataset': Split into train/val/test sets
                - 'create_dataloaders': Create dataloaders for each split
        
        Returns:
            A tuple containing:
              - A tuple of (train_loader, valid_loader, test_loader)
              - A tuple of (train_set, valid_set, test_set) tensors
              - A metadata dictionary
        """
        # Define the default pipeline if no steps provided
        if steps is None:
            steps = [
                'load_data',
                'data_truncation',
                'scaling',
                'delay_embedding',
                'weak_form',
                'create_dataset',
                'split_dataset',
                'create_dataloaders'
            ]
        
        # Define step mapping
        step_functions = {
            'load_data': lambda: self.load_data(self.data_path),
            'data_truncation': self.data_truncation,
            'scaling': self.apply_scaling,
            'delay_embedding': lambda: self.apply_delay_embedding() if self.model_type == 'NN' else None,
            'weak_form': lambda: self.generate_weak_form_params() if self.enable_weak_form else None,
            'create_dataset': self.create_dataset,
            'split_dataset': lambda: self.split_dataset(),
            'create_dataloaders': self.create_dataloaders
        }
        
        # Execute each step in order
        for step in steps:
            if step not in step_functions:
                raise ValueError(f"Unknown processing step: {step}")
            step_functions[step]()
        
        logging.info(f"Data processing complete. Train/Validation/Test sizes: {len(self.train_set)}, {len(self.valid_set)}, {len(self.test_set)}.")
        return (self.train_loader, self.valid_loader, self.test_loader), (self.train_set, self.valid_set, self.test_set), self.metadata

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Retrieve the dataset item at the specified index.
        
        Args:
            index (int): Index of the dataset item.
        
        Returns:
            torch.Tensor: The dataset entry corresponding to the index.
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Please call create_dataset() first.")
        return self.dataset[index]

    def __len__(self) -> int:
        """
        Return the total number of dataset entries.
        
        Returns:
            int: The number of dataset entries.
        """
        return 0 if self.dataset is None else len(self.dataset)