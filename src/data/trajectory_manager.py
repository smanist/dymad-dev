import os, logging, torch, yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as GeoDataLoader
from typing import Optional, Callable, Union, Tuple, Dict

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

    def __init__(self, config: Dict, device: torch.device = torch.device("cpu")):
        """
        Initialize the TrajectoryManager by loading the YAML config.
        
        Args:
            config (dict): Configuration dictionary.
            device (torch.device): Torch device to use.
        """
        self.config = config
        self.dtype = torch.double if config['data'].get('double_precision', False) else torch.float
        if self.config['data']['delay'] < 0:
            raise ValueError("Delay must be non-negative.")
        
        self.model_type = self.config['model']['type']
        self.enable_weak_form = self.config['weak_form']['enabled']
        self.device = device
        self.data_path = self.config['data']['path']
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
            self.metadata['dt'] = float(self.t)  # Record dt in metadata.
            self.t = [np.arange(traj.shape[0]) * self.metadata['dt'] for traj in self.x]
        elif isinstance(self.t, np.ndarray):
            if self.t.ndim == 2:
                # Single trajectory provided as a 2D array.
                self.metadata['dt'] = self.t[1] - self.t[0]
                self.t = [self.t]
            elif self.t.ndim == 3:
                self.metadata['dt'] = self.t[0][1] - self.t[0][0] ## TODO: this assumes same dt for all trajecotories
                self.t = [ti.squeeze() for ti in self.t]
            else:
                # Unsupported dimensions for t.
                raise ValueError(f"Unsupported array dimension for t: {self.t.ndim}")
        # Ensure that the lengths of x, t, and u match.
        if len(self.x) != len(self.t) or len(self.x) != len(self.u):
            raise ValueError("The lengths of x, t, and u must match.")

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

        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            self.x = self.x[:n_samples]
            self.u = self.u[:n_samples]
            self.t = self.t[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.x = [traj[:n_steps] for traj in self.x]
            # Apply truncation only if control inputs exist.
            self.u = [inp[:n_steps] if inp is not None else None for inp in self.u]
            self.t = [ti[:n_steps] for ti in self.t]

        # Update metadata.
        self.metadata["n_samples"] = len(self.x)
        self.metadata["n_steps"] = [len(traj) for traj in self.x]
        self.metadata["n_state_features"] = int(self.x[0].shape[-1])
        self.metadata["n_control_features"] = int(self.u[0].shape[-1]) if self.u[0].ndim > 1 else 0
        self.metadata["delay"] = delay
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
        scale_cfg = self.config.get("scaling", {})
        mode = scale_cfg.get("mode", "none")
        ckpt_path = scale_cfg.get("load_from_checkpoint", None)
        if ckpt_path not in (None, "None") and os.path.exists(ckpt_path):
            logging.info(f"Loading scaling parameters from checkpoint: {ckpt_path}.")
            # Load the checkpoint and extract scaler parameters.
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            scaler_params = checkpoint.get("scaler_parameters", None)
            if scaler_params is None:
                raise KeyError("Scaler parameters not found in checkpoint file.")
            self.off = scaler_params.get("offset", None)
            self.scl = scaler_params.get("scale", None)
            if self.off is None or self.scl is None:
                raise ValueError("Invalid scaler parameters in checkpoint.")
            # Initialize Scaler with externally provided parameters.
            self.scaler = Scaler(mode=mode, scl=self.scl, off=self.off)
        else:
            logging.info(f"Applying scaling with mode: {mode}.")
            # Create and fit a Scaler on the data.
            self.scaler = Scaler(mode=mode)
            self.scaler.fit(self.x)
            self.off = self.scaler._off
            self.scl = self.scaler._scl

        scale_cfg = self.config.get("scaling", {})
        mode = scale_cfg.get("mode", "none")
        file_path = scale_cfg.get("file", None)
        if file_path is None:
            raise ValueError("Scaler file path must be specified to save/load the scaler.")
        reload_flag = scale_cfg.get("reload", False)
        if reload_flag and os.path.exists(file_path):
            logging.info(f"Loading scaling parameters from file: {file_path}.")
            scaler_dict = np.load(file_path, allow_pickle=True)
            self.off = scaler_dict["offset"]
            self.scl = scaler_dict["scale"]
            if self.off is None or self.scl is None:
                raise ValueError("Invalid scaler parameters in file.")
            self.scaler = Scaler(mode=mode, scl=self.scl, off=self.off)
        else:
            logging.info(f"Applying scaling with mode: {mode}.")
            self.scaler = Scaler(mode=mode)
            self.scaler.fit(self.x)
            self.off = self.scaler._off
            self.scl = self.scaler._scl
            dir_name = os.path.dirname(file_path)
            os.makedirs(dir_name, exist_ok=True)
            np.savez(file_path, offset=self.off, scale=self.scl)

        # Transform the trajectories.
        self.x = self.scaler.transform(self.x)
        self.metadata['off'] = self.off
        self.metadata['scl'] = self.scl
        

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
        # If delay==0, nothing is changed.

    def generate_weak_form_params(self) -> None:
        """
        If weak form is enabled in the configuration, generate and store the weak form parameters.
        
        This method uses the generate_weak_weights function from the weak module.
        """
        logging.info("Generating weak form parameters.")
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
            tf=self.t[0][-1],  # TODO: non-uniform tf?
            n_steps=self.metadata["n_steps"][0], # TODO this needs to be updated
            n_integration_points=N,
            integration_stride=dN,
            poly_order=ordpol,
            int_rule_order=ordint,
            delay=self.metadata["delay"]
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
        # TODO: actually need to make this not a concatenated tensor, but keep X and U separate at this stage.


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
        
        Returns:
            train_set, valid_set, test_set.
        """
        split_cfg = self.config.get("split", {})
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

        train_set = dataset_rand[:n_train]
        valid_set = dataset_rand[n_train:n_train+n_val]
        test_set = dataset_rand[n_train+n_val:]
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
            features (torch.Tensor): Node feature matrix of shape (n_nodes, n_total_features) where n_nodes is the number of nodes.
            adj (torch.Tensor or np.ndarray, optional): A square adjacency matrix of shape (n_nodes, n_nodes). 
                Nonzero entries indicate edges. Their values are used as edge attributes.
            custom_fn (callable, optional): A custom function that takes (traj, adj) and returns a PyGData object.
                If provided, this function overrides the default graph construction.
        
        Returns:
            PyGData: A PyTorch Geometric Data object containing:
                - x: node n_total_features (same as traj)
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
                adj = torch.tensor(adj, dtype=self.dtype, device=features.device)
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

    def create_dataloader(
        self, dataset: torch.Tensor, shuffle: bool = True
    ) -> DataLoader:
        """
        Create a dataloader for a single dataset based on the model type.
        
        The model type is specified in the YAML config under "dataloader/model_type"
        and can be one of "NN", "LSTM", or "GNN".
        
        Args:
            dataset (torch.Tensor): The dataset to load.
            shuffle (bool): Whether to shuffle the dataset.
        
        Returns:
            DataLoader: The dataloader for the dataset.
        """
        dl_cfg = self.config.get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)

        if self.model_type == "NN":
            logging.info(f"Creating dataloader for NN model with batch size {batch_size}.")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        elif self.model_type == "LSTM":
            logging.info(f"Creating dataloader for LSTM model with batch size {batch_size}.")
            X, y = self._create_sequences(dataset)
            loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

        elif self.model_type == "GNN":
            logging.info(f"Creating dataloader for GNN model with batch size {batch_size}.")
            gnn_cfg = dl_cfg.get("gnn", {})
            adj = gnn_cfg.get("adjacency", None)
            custom_fn = gnn_cfg.get("custom_fn", None)
            # Build graph objects for each trajectory sample.
            data_list = [self._create_graph(sample, adj=adj, custom_fn=custom_fn) for sample in dataset]
            loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return loader

    def _create_sequences(
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

    def process_all(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
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
        self.load_data(self.data_path)
        self.data_truncation()
        self.apply_scaling()
        if self.model_type != 'LSTM':
            self.apply_delay_embedding()
        if self.enable_weak_form:
            self.generate_weak_form_params()
        self.create_dataset()
        splited_datasets = self.split_dataset()
        shuffle = [True, False, False]
        loaders = [self.create_dataloader(ds, sh) for ds, sh in zip(splited_datasets, shuffle)]
        logging.info(f"Data processing complete. Train/Validation/Test sizes: {len(splited_datasets[0])}, {len(splited_datasets[1])}, {len(splited_datasets[2])}.")
        return loaders, splited_datasets, self.metadata

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