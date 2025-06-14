import os, logging, torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as GeoDataLoader
from typing import Optional, Union, Tuple, Dict, List

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
        self.model_type = self.metadata['config']['model']['type'].upper()
        self.enable_weak_form = self.metadata['config']['weak_form']['enabled']
        self.device = device
        self.data_path = self.metadata['config']['data']['path']
        self.adj = adj  # Store the adjacency matrix if provided externally
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
        logging.info(f"x shape: {self.x.shape if isinstance(self.x, np.ndarray) else f'{len(self.x)} list of arrays'}")
        logging.info(f"t shape: {self.t.shape if isinstance(self.t, np.ndarray) else f'{len(self.t)} list of arrays'}")
        logging.info(f"u shape: {self.u.shape if isinstance(self.u, np.ndarray) else f'{len(self.u)} list of arrays'}")
        # Process x
        if isinstance(self.x, np.ndarray):
            if self.x.ndim == 3:  # multiple trajectories as (n_traj, n_steps, n_features)
                logging.info(f"Detected x as 3D np.ndarray (n_traj, n_steps, n_features): {self.x.shape}. Splitting into list of arrays.")
                self.x = [np.array(_x) for _x in self.x]
            elif self.x.ndim == 2:  # single trajectory (n_steps, n_features)
                logging.info(f"Detected x as 2D np.ndarray, treating it as a single trajectory (n_steps, n_features): {self.x.shape}. Wrapping as single-element list.")
                self.x = [np.array(self.x)]
            else:
                logging.error(f"Unsupported x shape: {self.x.shape}")
                raise ValueError(f"Unsupported x shape: {self.x.shape}")
        elif isinstance(self.x, list):
            logging.info(f"Detected x as list of arrays.")
            self.x = [np.array(_x) for _x in self.x]
        else:
            logging.error("x must be a numpy array or list of arrays")
            raise TypeError("x must be a numpy array or list of arrays")

        # Process u
        if self.u is None:
            logging.info("No control input u detected. Creating zero-valued control inputs for autonomous system.")
            # Create zero control inputs with shape (n_steps, 1) for each trajectory
            self.u = [np.zeros((x.shape[0], 1)) for x in self.x]
        elif isinstance(self.u, np.ndarray):
            if self.u.ndim == 3:  # (n_traj, n_steps, n_controls)
                logging.info(f"Detected u as 3D np.ndarray (n_traj, n_steps, n_controls): {self.u.shape}. Splitting into list of arrays.")
                self.u = [np.array(_u) for _u in self.u]
            elif self.u.ndim == 2:  # (n_steps, n_controls)
                if len(self.x) > 1:
                    logging.info(f"Detected u as 2D np.ndarray (n_steps, n_controls) but x is multi-traj ({len(self.x)}). Broadcasting u to all trajectories.")
                    self.u = [np.array(self.u) for _ in self.x]
                else:
                    logging.info(f"Detected u as 2D np.ndarray (n_steps, n_controls): {self.u.shape}. Wrapping as single-element list.")
                    self.u = [np.array(self.u)]
            elif self.u.ndim == 1:  # (n_controls,) - constant control input
                logging.info(f"Detected u as 1D np.ndarray (n_controls,): {self.u.shape}. Expanding to trajectory for each x and broadcasting to all trajectories.")
                self.u = [np.tile(self.u, (x.shape[0], 1)) for x in self.x]
            else:
                logging.error(f"Unsupported u shape: {self.u.shape}")
                raise ValueError(f"Unsupported u shape: {self.u.shape}")
        elif isinstance(self.u, list):
            if len(self.u) == 1 and len(self.x) > 1:
                # Single u for multiple x, broadcast
                u0 = np.array(self.u[0])
                if u0.ndim == 1:
                    logging.info(f"Detected u as single 1D array in list for multiple x. Expanding and broadcasting to all trajectories.")
                    self.u = [np.tile(u0, (x.shape[0], 1)) for x in self.x]
                elif u0.ndim == 2:
                    logging.info(f"Detected u as single 2D array in list for multiple x. Broadcasting to all trajectories.")
                    self.u = [np.array(u0) for _ in self.x]
                else:
                    logging.error(f"Unsupported u shape in list: {u0.shape}")
                    raise ValueError(f"Unsupported u shape in list: {u0.shape}")
            else:
                logging.info(f"Detected u as list of arrays. Converting all to np.ndarray.")
                self.u = [np.array(_u) for _u in self.u]
        else:
            logging.error("u must be a numpy array or list of arrays")
            raise TypeError("u must be a numpy array or list of arrays")

        # Ensure x and u have matching trajectory count and length
        if len(self.x) != len(self.u):
            logging.error("x and u must have the same number of trajectories")
            raise ValueError("x and u must have the same number of trajectories")
        for xi, ui in zip(self.x, self.u):
            if xi.shape[0] != ui.shape[0]:
                logging.error("Each trajectory in x and u must have the same number of time steps")
                raise ValueError("Each trajectory in x and u must have the same number of time steps")

        # Process t
        if isinstance(self.t, (float, int)):
            logging.info(f"Detected t as scalar ({self.t}). Generating uniform time arrays for each trajectory.")
            self.dt = float(self.t)
            self.t = [np.arange(traj.shape[0]) * self.dt for traj in self.x]
            self.dt = [self.dt for _ in self.t]
        elif isinstance(self.t, np.ndarray):
            if self.t.ndim == 1:
                if len(self.x) > 1:
                    logging.info(f"Detected t as 1D np.ndarray (n_steps,) but x is multi-traj ({len(self.x)}). Broadcasting t to all trajectories.")
                    self.t = [np.array(self.t) for _ in self.x]
                    self.dt = [self.t[0][1] - self.t[0][0] for _ in self.x]
                else:
                    logging.info(f"Detected t as 1D np.ndarray (n_steps,): {self.t.shape}. Wrapping as single-element list.")
                    self.dt = [self.t[1] - self.t[0]]
                    self.t = [self.t]
            elif self.t.ndim == 2:
                logging.info(f"Detected t as 2D np.ndarray (n_traj, n_steps): {self.t.shape}. Splitting into list of arrays.")
                self.t = [ti for ti in self.t]
                self.dt = [ti[1] - ti[0] for ti in self.t]
            else:
                logging.error(f"Unsupported array dimension for t: {self.t.ndim}")
                raise ValueError(f"Unsupported array dimension for t: {self.t.ndim}")
        elif isinstance(self.t, list):
            if len(self.t) == 1 and len(self.x) > 1:
                logging.info(f"Detected t as single array in list for multiple x. Broadcasting t to all trajectories.")
                t0 = np.array(self.t[0])
                self.t = [t0 for _ in self.x]
                self.dt = [t0[1] - t0[0] for _ in self.x]
            else:
                logging.info(f"Detected t as list of arrays. Converting all to np.ndarray and computing dt for each.")
                self.t = [np.array(ti) for ti in self.t]
                self.dt = [ti[1] - ti[0] for ti in self.t]
        else:
            logging.error("t must be a float, numpy array, or list of numpy arrays")
            raise TypeError("t must be a float, numpy array, or list of numpy arrays")
        
        # Try to load adjacency matrix from data if not provided externally and model type is GNN
        if self.adj is None and self.model_type == "GNN":
            try:
                self.adj = data['adj_mat']
                logging.info("Loaded adjacency matrix from data file")
            except KeyError:
                logging.error("No adjacency matrix found in data file and none provided externally")
                raise ValueError("Adjacency matrix is required for GNN model type but none was found")

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
            # Truncate control inputs to match trajectory length
            self.u = [inp[:n_steps] for inp in self.u]
            self.t = [ti[:n_steps] for ti in self.t]
        
        # Store dt and n_steps metadata
        self.metadata["dt_and_n_steps"] = self._create_dt_n_steps_metadata()

        # Populate metadata.
        self.metadata['delay'] = self.delay
        self.metadata["n_samples"] = len(self.x)
        self.metadata["n_state_features"] = int(self.x[0].shape[-1])
        # For autonomous systems, we created zero controls with 1 feature, but logically it's 0 controls
        n_control_features = int(self.u[0].shape[-1])
        # Check if this is an autonomous system (all controls are zero)
        is_autonomous = all(np.allclose(u, 0) for u in self.u)
        self.metadata["n_control_features"] = 0 if is_autonomous and n_control_features == 1 else n_control_features
        logging.info("Data loaded and processed.")
        logging.info(f"Number of samples: {self.metadata['n_samples']}")
        logging.info(f"Number of state features: {self.metadata['n_state_features']}")
        logging.info(f"Number of control features: {self.metadata['n_control_features']}")
        logging.info(f"Delay embedding size: {self.metadata['delay']}")

    def _create_dt_n_steps_metadata(self) -> List[List[float]]:
        """
        Create metadata for dt and n_steps, optimizing storage if values are uniform.
        
        Returns:
            List of [dt, n_steps] pairs. If all trajectories have the same dt and n_steps,
            returns only one entry for optimization.
        """
        # Store dt and n_steps for metadata, but don't modify self.t and self.dt
        metadata_dt_and_n_steps = []
        for dt, t in zip(self.dt, self.t):
            # Use the actual length after any truncation for metadata
            actual_n_steps = len(t)
            metadata_dt_and_n_steps.append([dt, actual_n_steps])
        
        # Check if uniform dt and n_steps for metadata optimization
        if len(metadata_dt_and_n_steps) > 0:
            dts = [item[0] for item in metadata_dt_and_n_steps]
            nsteps = [item[1] for item in metadata_dt_and_n_steps]
            if len(set(dts)) == 1 and len(set(nsteps)) == 1:
                # Only store one entry if both dt and n_steps are uniform
                logging.info("Uniform dt and n_steps detected across all trajectories. Only saving one entry in metadata.")
                return [metadata_dt_and_n_steps[0]]
            else:
                return metadata_dt_and_n_steps
        else:
            return []

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
            self.metadata["dt_and_n_steps"] = self._create_dt_n_steps_metadata()
            self.metadata
        # If delay==0, nothing is changed.

    def generate_weak_form_params(self) -> None:
        """
        If weak form is enabled in the configuration, generate and store the weak form parameters.
        
        This method uses the generate_weak_weights function from the weak module.
        """
        weak_cfg = self.metadata['config'].get("weak_form", {})
        if not weak_cfg.get("enabled", False):
            return

        logging.info("Generating weak form parameters with the following configuration:")
        logging.info(weak_cfg)

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

        if len(self.metadata["dt_and_n_steps"]) > 1:
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
        self.metadata["weak_dyn_param"] = self.weak_dyn_param
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
        self.metadata['n_total_state_features'] = self.metadata['n_state_features']*(self.metadata['delay']+1)
        self.metadata['n_total_features'] = self.metadata['n_total_state_features'] + self.metadata['n_control_features']
        
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
            train_X, train_y = self._create_lstm_sequences(self.train_set)
            valid_X, valid_y = self._create_lstm_sequences(self.valid_set)
            test_X, test_y = self._create_lstm_sequences(self.test_set)
            
            # self.train_set = TensorDataset(train_X, train_y)
            # self.valid_set = TensorDataset(valid_X, valid_y)
            # self.test_set = TensorDataset(test_X, test_y)
            
            # Create dataloaders
            self.train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
            self.valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

        elif self.model_type == "GNN":
            logging.info(f"Creating dataloaders for GNN model with batch size {batch_size}.")
            gnn_cfg = dl_cfg.get("gnn", {})
            # Use provided adj matrix if available, otherwise try to get from config (TODO: does not support dynamic graphs yet)
            adj = self.adj if self.adj is not None else gnn_cfg.get("adjacency", None)
            
            # Build graph objects for each set
            self.train_set = self._create_gnn_sequences(self.train_set, adj)
            self.valid_set = self._create_gnn_sequences(self.valid_set, adj)
            self.test_set = self._create_gnn_sequences(self.test_set, adj)
            self.t = [ti[:-(self.metadata['delay']+1)] for ti in self.t]
            self.t = [self.t[0]] if len(set(map(len, self.t))) == 1 else self.t
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
        # Update metadata
        self.metadata['n_features_per_node'] = seq_length*n_features_per_node
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