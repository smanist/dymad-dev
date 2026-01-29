import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple, Dict, List

from dymad.io.data import DynData
from dymad.transform import make_transform
from dymad.utils.graph import adj_to_edge

logger = logging.getLogger("dymad.cv")

def _process_data(data, x, label, base_dim=1, offset=0):
    """
    x as reference data, list of arrays.

    When offset = 1, effectively the n_steps dimension is removed, so the method processes time-invariant data.

    Expecting to return a list of `n_traj` arrays:
    _data = [... d_i ...]
    where d_i has shape (n_steps, ...) and ndim(d_i) = base_dim + 1

    d_i itself can be an array or a list, depending on the input data type.
    """
    _dim = base_dim - offset
    if data is None:
        logger.info(f"No {label} detected. Setting to None.")
        if offset == 0:
            _data = [np.empty((_x.shape[0], 0)) for _x in x]
        else:
            _data = [np.empty((0,)) for _ in x]

    elif isinstance(data, np.ndarray):
        # t,y,u,p should always go through here, by converting to np.ndarray beforehand.
        # ei/ew/ea could go through this branch too, if they are in np.ndarray form;
        # this would mean their shapes are uniform throughout the dataset.
        #
        # The possibilities here:
        # - np.ndarray of shape (n_traj, n_steps, ...)  - full data for multiple trajs
        # - np.ndarray of shape (1, n_steps, ...) - Broadcast to all trajs if needed
        # - np.ndarray of shape (n_traj, 1, ...)  - Broadcast to all steps
        # - np.ndarray of shape (n_steps, ...)    - single traj data, broadcast to all trajs if needed
        # - np.ndarray of shape (...,)            - Broadcast to all steps and trajs
        if data.ndim == _dim + 2:  # (n_traj, n_steps, ...)
            if data.shape[0] == 1 and len(x) > 1:
                logger.info(f"Detected {label} as np.ndarray (1, n_steps, ...): {data.shape} for multiple x. Broadcasting to all trajectories.")
                _data = [np.array(data[0]) for _ in x]
            elif data.shape[1] == 1 and offset == 0:   # Need offset == 0, otherwise n_steps dimension is irrelevant (for p, offset=1)
                logger.info(f"Detected {label} as np.ndarray (n_traj, 1, ...): {data.shape}. Expanding to trajectory for each x and broadcasting to all time steps.")
                _data = [np.tile(data[_i], (_x.shape[0],) + (1,) * base_dim) for _i, _x in enumerate(x)]
            else:
                logger.info(f"Detected {label} as np.ndarray (n_traj, n_steps, ...): {data.shape}. Splitting into list of arrays.")
                _data = [np.array(_u) for _u in data]
        elif data.ndim == _dim + 1:  # (n_steps, ...)
            if len(x) > 1:
                logger.info(f"Detected {label} as np.ndarray (n_steps, ...): {data.shape} but x is multi-traj ({len(x)}). Broadcasting {label} to all trajectories.")
                _data = [np.array(data) for _ in x]
            else:
                logger.info(f"Detected {label} as np.ndarray (n_steps, ...): {data.shape}. Wrapping as single-element list.")
                _data = [np.array(data)]
        elif data.ndim == _dim and _dim > 0:  # (...,)
            logger.info(f"Detected {label} as np.ndarray (...,): {data.shape}. Expanding to trajectory for each x and broadcasting to all trajectories.")
            _data = [np.tile(data, (x.shape[0],) + (1,) * base_dim) for x in x]
        else:
            msg = f"Unsupported {label} shape: {data.shape}"
            logger.error(msg)
            raise ValueError(msg)

    elif isinstance(data, list):
        # This branch should be ei/ew/ea that are in lists.
        # an element of np.ndarray is considered one sample at one step in one trajectory,
        # so its ndim should be base_dim.
        #
        # The possibilities here:
        # - list of lists of arrays    - (n_traj, n_steps, ...) - already full data
        # - list of one list of arrays - (1, n_steps, ...) - broadcast to all trajs
        # - list of lists of one array - (n_traj, 1, ...)  - broadcast to all steps
        # - list of arrays             - (n_steps, ...)    - single traj data, broadcast to all trajs
        if isinstance(data[0], np.ndarray):
            if data[0].ndim == _dim:   # (n_steps, ...)
                if len(x) > 1:
                    logger.info(f"Detected {label} as lists (n_steps, ...): {data[0].shape} but x is multi-traj ({len(x)})." \
                                 f"Broadcasting {label} to all trajectories.")
                    _data = [data for _ in x]
                else:
                    logger.info(f"Detected {label} as lists (n_steps, ...): {data[0].shape}. Wrapping as single-element list.")
                    _data = [np.array(data[0])]
            else:
                msg = f"Unsupported {label} array shape in list: {data[0].shape}"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(data[0], list):
            if len(data) == 1:         # (1, n_steps, ...)
                if len(x) > 1:
                    logger.info(f"Detected {label} as lists (1, n_steps, ...): {data[0][0].shape} for multiple x. Broadcasting to all trajectories.")
                    _data = [data[0] for _ in x]
                else:
                    logger.info(f"Detected {label} as lists (n_traj, n_steps, ...): {data[0][0].shape}. Return as is.")
                    _data = data
            elif len(data[0]) == 1:    # (n_traj, 1, ...)
                logger.info(f"Detected {label} as lists (n_traj, 1, ...): {data[0][0].shape}. Expanding to trajectory for each x and broadcasting to all time steps.")
                _data = [[data[_i][0] for _ in range(x[_i].shape[0])] for _i in range(len(x))]
            else:                      # (n_traj, n_steps, ...)
                logger.info(f"Detected {label} as lists (n_traj, n_steps, ...): {data[0][0].shape}. Return as is.")
                _data = data

    else:
        logger.error(f"{label} must be a np.ndarray or list of np.ndarrays")
        raise TypeError(f"{label} must be a np.ndarray or list of np.ndarrays")

    # Data validation
    assert len(_data) == len(x), f"{label} list length ({len(_data)}) must match x list length ({len(x)})"
    if len(_data[0]) > 0 and offset == 0:
        for xi, ui in zip(x, _data):
            if len(xi) != len(ui):
                msg = f"Each trajectory in x ({len(xi)}) and {label} ({len(ui)}) must have the same number of time steps"
                logger.error(msg)
                raise ValueError(msg)
    return _data

class TrajectoryManager:
    """
    A class to manage trajectory data loading, preprocessing, and
    dataloader creation.

    The workflow includes:

      - Loading raw data from a binary file.
      - Preprocessing (trimming trajectories, subsetting, etc.).
      - Creating a dataset.
      - Normalizing and transforming the data using specified transformations.
      - Creating a dataloader.

    The class is configured via a YAML configuration file.

    Args:
        metadata (dict): Configuration dictionary.
        mode (str): Dataset to read, one of 'train', 'valid', 'test'.
        device (torch.device): Torch device to use.
    """

    # --------------
    # Initialization
    # --------------
    def __init__(
            self,
            metadata: Dict,
            data_key: str | None = None,
            device: torch.device = torch.device("cpu")):
        self.metadata = copy.deepcopy(metadata)
        self.device = device

        self._init_transforms()
        self._load_metadata(self.metadata, data_key)

    def _init_transforms(self) -> None:
        self._transform_fitted = False
        self._data_transform_x = make_transform(self.metadata['config'].get('transform_x', None))
        self._data_transform_y = make_transform(self.metadata['config'].get('transform_y', None))
        self._data_transform_p = make_transform(self.metadata['config'].get('transform_p', None))
        cfg_transform_u = self.metadata['config'].get('transform_u', None)
        self._data_transform_u = make_transform(cfg_transform_u)
        if cfg_transform_u is None:
            self.metadata["delay"] = self._data_transform_x.delay
        else:
            self.metadata["delay"] = max(self._data_transform_x.delay, self._data_transform_u.delay)

    def _load_metadata(self, metadata: Dict, data_key: str) -> None:
        if "data_key" in metadata:
            self.data_key = metadata["data_key"]
        else:
            if data_key == 'train':
                self.data_key = 'data'
            else:
                self.data_key = 'data_' + data_key
            self.metadata["data_key"] = self.data_key
        self.data_path = self.metadata['config'][self.data_key]['path']
        self.dtype = torch.double if self.metadata['config'][self.data_key].get('double_precision', False) else torch.float

        if "data_index" in metadata:
            # If data_index is already in metadata, we assume the dataset has been processed before.
            assert metadata["n_data"] == len(metadata["data_index"])
            logger.info(f"Reusing data index from provided metadata.")

            self.data_index = torch.tensor(metadata["data_index"], dtype=torch.long)
            self.metadata["n_data"] = metadata["n_data"]
            self.metadata["data_index"] = self.data_index.tolist()

            self.set_transforms(metadata=metadata)    # This sets self._transform_fitted = True
        else:
            self.data_index = None
            self._transform_fitted = False

    # --------------
    # Public interface - for modification
    # --------------
    def update_config(self, config: Dict) -> None:
        """
        Update the configuration metadata.
        After this step, data transformations need to be refitted.
        """
        self.metadata['config'].update(config)
        self._init_transforms()
        logger.info("New config loaded.")

    def set_transforms(
            self,
            metadata: Dict | None = None,
            trajmgr: Optional['TrajectoryManager'] = None
            ) -> None:
        if (metadata is None and trajmgr is None) or (metadata is not None and trajmgr is not None):
            raise ValueError("Either metadata or trajmgr must be provided, but not both.")

        if metadata is not None:
            self._data_transform_x.load_state_dict(metadata["transform_x_state"])
            if "transform_y_state" in metadata:
                self._data_transform_y.load_state_dict(metadata["transform_y_state"])
            if "transform_u_state" in metadata:
                self._data_transform_u.load_state_dict(metadata["transform_u_state"])
            if "transform_p_state" in metadata:
                self._data_transform_p.load_state_dict(metadata["transform_p_state"])
        else:
            self._data_transform_x.load_state_dict(trajmgr._data_transform_x.state_dict())
            if hasattr(trajmgr, "_data_transform_y") and trajmgr._data_transform_y is not None:
                self._data_transform_y.load_state_dict(trajmgr._data_transform_y.state_dict())
            if hasattr(trajmgr, "_data_transform_u") and trajmgr._data_transform_u is not None:
                self._data_transform_u.load_state_dict(trajmgr._data_transform_u.state_dict())
            if hasattr(trajmgr, "_data_transform_p") and trajmgr._data_transform_p is not None:
                self._data_transform_p.load_state_dict(trajmgr._data_transform_p.state_dict())
        self.metadata["transform_x_state"] = self._data_transform_x.state_dict()
        self.metadata["transform_y_state"] = self._data_transform_y.state_dict() if self._data_transform_y is not None else None
        self.metadata["transform_u_state"] = self._data_transform_u.state_dict() if self._data_transform_u is not None else None
        self.metadata["transform_p_state"] = self._data_transform_p.state_dict() if self._data_transform_p is not None else None
        self._transform_fitted = True

    def set_data_index(self, index: Union[torch.Tensor, List[int]] | None = None) -> None:
        """
        Set the data index for this TrajectoryManager.
        """
        if index is None:
            # By default use all data
            self.data_index = torch.arange(0, len(self.x), dtype=torch.long)
        else:
            if isinstance(index, list):
                index = torch.tensor(index, dtype=torch.long)
            self.data_index = index

        self.metadata["n_data"] = len(self.data_index)
        self.metadata["data_index"] = self.data_index.tolist()

        logger.info(f"Data index set: {self.metadata['n_data']} trajectories.")

    # --------------
    # Public interface - for workflow
    # --------------
    def prepare_data(self) -> None:
        """
        Handy function to load and truncate data in one call.
        """
        self.load_data()
        self.data_truncation()

    def process_data(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Latter half of process_all
        """
        self.apply_data_transformations()
        self.create_dataloaders()

        logger.info(f"Data processing complete. Data size: {len(self.dataset)}.")
        return self.dataloader, self.dataset, self.metadata

    def process_all(self) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Returns:
            A tuple containing: dataloader, dataset, metadata
        """
        self.prepare_data()
        if self.data_index is None:
            self.set_data_index()
        res = self.process_data()
        return res

    # --------------
    # Workflow implementation - not meant for public use
    # --------------
    def load_data(self) -> Dict:
        """
        Load raw data from a binary file.

        The file is assumed to store (in order):
            x: array-like or list of array-like, shape (n_samples, n_state_features)
            data. If data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

            t: float, numpy array of shape (n_samples,), or list of numpy arrays
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time (seconds in physical time) at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.

            u: array-like or list of array-like, shape (n_samples, n_control_features), optional (default None)
            Control variables/inputs.
            If data contains multiple trajectories (i.e. if x is a list of
            array-like), then u should be a list containing control variable data
            for each trajectory. Individual trajectories may contain different
            numbers of samples.
        """
        # Load the binary data from the file.
        data = np.load(self.data_path, allow_pickle=True)

        # Extract entries from the loaded data.
        logger.info("Loading raw data...")
        keys = ['t', 'x', 'y', 'u', 'p']
        vals = []
        for k in keys:
            _tmp = data.get(k, None)
            if k == 'x' and _tmp is None:
                msg = "x must be provided in the data file."
                logger.error(msg)
                raise ValueError(msg)
            if _tmp is not None:
                logger.info(f"{k} shape: {_tmp.shape if isinstance(_tmp, np.ndarray) else f'{len(_tmp)} list of arrays'}")
            vals.append(_tmp)
        logger.info("Raw data loaded.")

        # Process x
        x = vals[1]
        if isinstance(x, np.ndarray):
            if x.ndim == 3:  # multiple trajectories as (n_traj, n_steps, n_features)
                logger.info(f"Detected x as 3D np.ndarray (n_traj, n_steps, n_features): {x.shape}. Splitting into list of arrays.")
                self.x = [np.array(_x) for _x in x]
            elif x.ndim == 2:  # single trajectory (n_steps, n_features)
                logger.info(f"Detected x as 2D np.ndarray, treating it as a single trajectory (n_steps, n_features): {x.shape}. Wrapping as single-element list.")
                self.x = [np.array(x)]
            else:
                msg = f"Unsupported x shape: {x.shape}"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(x, list):
            logger.info(f"Detected x as list of arrays.")
            self.x = [np.array(_x) for _x in x]
        else:
            logger.error("x must be a np.ndarray or list of np.ndarrays")
            raise TypeError("x must be a np.ndarray or list of np.ndarrays")

        # In the processing below, the raw data is converted to arrays, as they are supposed to be regular.
        # Process t
        self.t = _process_data(
            None if vals[0] is None else np.array(vals[0]),
            self.x, "t", base_dim=0, offset=0)
        if self.t[0].size == 0:
            self.t = [np.arange(_x.shape[0]) for _x in self.x]
        self.dt = [ti[1] - ti[0] for ti in self.t]

        # Process y
        self.y = _process_data(
            None if vals[2] is None else np.array(vals[2]),
            self.x, "y", base_dim=1, offset=0)

        # Process u
        self.u = _process_data(
            None if vals[3] is None else np.array(vals[3]),
            self.x, "u", base_dim=1, offset=0)
        self._is_autonomous = self.u[0].size == 0

        # Process p
        self.p = _process_data(
            None if vals[4] is None else np.array(vals[4]),
            self.x, "p", base_dim=1, offset=1)
        
        return data

    def data_truncation(self) -> None:
        """
        Truncate the loaded data according to the configuration.

        This includes:
          - Subsetting the number of trajectories and horizon (n_steps).
          - Populating basic metadata (dt, tf, shapes, etc.).
        """
        cfg = self.metadata['config'].get(self.data_key, {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)

        if self.x is None or self.u is None or self.t is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            if n_samples > 1:
                self.t = self.t[:n_samples]
                self.dt = self.dt[:n_samples]
                self.x = self.x[:n_samples]
                self.y = self.y[:n_samples]
                self.u = self.u[:n_samples]
                self.p = self.p[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.t = [_t[:n_steps] for _t in self.t]
            self.x = [_x[:n_steps] for _x in self.x]
            self.y = [_y[:n_steps] for _y in self.y]
            self.u = [_u[:n_steps] for _u in self.u]
            # self.p is time-invariant

        # Populate metadata.
        self.metadata["n_samples"] = len(self.x)
        self.metadata["n_state_features"] = int(self.x[0].shape[-1])
        self.metadata["n_aux_features"] = int(self.y[0].shape[-1])
        self.metadata["n_control_features"] = int(self.u[0].shape[-1])
        self.metadata["n_parameters"] = int(self.p[0].shape[-1])
        logger.info("Data loaded and processed.")
        logger.info(f"Number of samples: {self.metadata['n_samples']}")
        logger.info(f"Number of state features: {self.metadata['n_state_features']}")
        logger.info(f"Number of auxiliary features: {self.metadata['n_aux_features']}")
        logger.info(f"Number of control features: {self.metadata['n_control_features']}")
        logger.info(f"Number of parameters: {self.metadata['n_parameters']}")
        logger.info(f"Delay embedding size: {self.metadata['delay']}")

    def apply_data_transformations(self) -> None:
        """
        Apply data transformations to the loaded trajectories and control inputs.
        This creates the dataset.

        This method applies transformations defined in the configuration for x, y, u, p
        """
        assert self.data_index is not None, "Dataset must be split before applying transformations."
        if not self._transform_fitted:
            logger.info("Fitting transformation for state features.")
            X = [self.x[i] for i in self.data_index]
            self._data_transform_x.fit(X)
            self.metadata["transform_x_state"] = self._data_transform_x.state_dict()

            if self.metadata["n_aux_features"] > 0:
                logger.info("Fitting transformation for auxiliary features.")
                Y = [self.y[i] for i in self.data_index]
                self._data_transform_y.fit(Y)
                self.metadata["transform_y_state"] = self._data_transform_y.state_dict()

            if self.metadata["n_control_features"] > 0:
                logger.info("Fitting transformation for control inputs.")
                U = [self.u[i] for i in self.data_index]
                self._data_transform_u.fit(U)
                self.metadata["transform_u_state"] = self._data_transform_u.state_dict()

            if self.metadata["n_parameters"] > 0:
                logger.info("Fitting transformation for parameters.")
                P = [self.p[i] for i in self.data_index]
                self._data_transform_p.fit(P)
                self.metadata["transform_p_state"] = self._data_transform_p.state_dict()
        else:
            logger.info("Transformations already fitted. Skipping fitting step.")

        logger.info("Applying transformations to state features and control inputs.")
        self.dataset = self._transform_by_index(self.data_index)

        if self.metadata["delay"] > 0:
            logger.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        # Process X first
        # If the common delay is larger (larger delay in u), we trim out the first few steps.
        # This way the latest x and u are aligned.
        _X = self._data_transform_x.transform([self.x[i] for i in indices])
        _d = self.metadata["delay"] - self._data_transform_x.delay
        if _d > 0:
            _X = [x[_d:] for x in _X]

        _T = [self.t[i] for i in indices]
        if self.metadata["delay"] > 0:
            _T = [t[self.metadata["delay"]:] for t in _T]

        if self.metadata["n_aux_features"] > 0:
            # Process Y only if there are auxiliary features.
            # Same idea as for X, we trim out the first few steps if the delay in x is larger.
            _Y = self._data_transform_y.transform([self.y[i] for i in indices])
            _d = self.metadata["delay"] - self._data_transform_y.delay
            if _d > 0:
                _Y = [y[_d:] for y in _Y]
        else:
            _Y = [None for _ in _X]

        if self.metadata["n_control_features"] > 0:
            # Process U only if there are control features.
            # Same idea as for X, we trim out the first few steps if the delay in x is larger.
            _U = self._data_transform_u.transform([self.u[i] for i in indices])
            _d = self.metadata["delay"] - self._data_transform_u.delay
            if _d > 0:
                _U = [u[_d:] for u in _U]
        else:
            _U = [None for _ in _X]

        if self.metadata["n_parameters"] > 0:
            # Nothing to delay for parameters, they are time-invariant.
            _P = self._data_transform_p.transform([self.p[i] for i in indices])
        else:
            _P = [None for _ in _X]

        # Lastly assemble the dataset.
        dataset = []
        for _t, _x, _y, _u, _p in zip(_T, _X, _Y, _U, _P):
            dataset.append(DynData(
                t=torch.tensor(_t, dtype=self.dtype, device=self.device),
                x=torch.tensor(_x, dtype=self.dtype, device=self.device),
                y=torch.tensor(_y, dtype=self.dtype, device=self.device) if _y is not None else None,
                u=torch.tensor(_u, dtype=self.dtype, device=self.device) if _u is not None else None,
                p=torch.tensor(_p, dtype=self.dtype, device=self.device) if _p is not None else None
        ))
        return dataset

    def _update_dataset_metadata(self):
        # Bookkeeping metadata for the dataset.
        self.metadata['n_total_state_features'] = self._data_transform_x._out_dim
        if self.metadata["n_aux_features"] == 0:
            self.metadata['n_total_aux_features'] = 0
        else:
            self.metadata['n_total_aux_features'] = self._data_transform_y._out_dim
        if self.metadata["n_control_features"] == 0:
            self.metadata['n_total_control_features'] = 0
        else:
            self.metadata['n_total_control_features'] = self._data_transform_u._out_dim
        if self.metadata["n_parameters"] == 0:
            self.metadata['n_total_parameters'] = 0
        else:
            self.metadata['n_total_parameters'] = self._data_transform_p._out_dim
        self.metadata['n_total_features'] = self.metadata['n_total_state_features'] + self.metadata['n_total_control_features']
        self.metadata["dt_and_n_steps"] = self._create_dt_n_steps_metadata()

        logger.info(f"Number of total state features: {self.metadata['n_total_state_features']}")
        logger.info(f"Number of total auxiliary features: {self.metadata['n_total_aux_features']}")
        logger.info(f"Number of total control features: {self.metadata['n_total_control_features']}")
        logger.info(f"Number of total parameters: {self.metadata['n_total_parameters']}")

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
                logger.info("Uniform dt and n_steps detected across all trajectories. Only saving one entry in metadata.")
                return [metadata_dt_and_n_steps[0]]
            else:
                return metadata_dt_and_n_steps
        else:
            return []

    def create_dataloaders(self) -> None:
        """
        Create dataloaders for the data set.
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)
        if_shuffle: bool = dl_cfg.get("shuffle", True)

        logger.info(f"Creating dataloaders for model with batch size {batch_size}.")
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=if_shuffle, collate_fn=DynData.collate)

class TrajectoryManagerGraph(TrajectoryManager):
    r"""
    A class to manage trajectory data loading, preprocessing, and
    dataloader creation - graph version.

    The graph data is assumed to be homogeneous, that each node has the same number of features.
    Hence the normalization, if done, is applied globally to all nodes.

    However, the number of edges can vary over time, and hence other quantities defined on edges.

    In the raw data, the nodal state features are expected to be concatenated sequentially.
    For example, for N nodes with M features each, the raw data for states at a time step is
    
    .. math::
        x = [x_1, x_2, ..., x_N], \text{where } x_i \in R^M,

    Same applies to other data members, if present.

    Args:
        metadata (dict): Configuration dictionary.
        device (torch.device): Torch device to use.
        adj (torch.Tensor or np.ndarray, optional): Adjacency matrix for GNN models.
            If not provided, will try to get from config.
    """

    # --------------
    # Initialization
    # --------------
    def __init__(
            self,
            metadata: Dict,
            data_key: str = 'train',
            device: torch.device = torch.device("cpu"),
            adj: Optional[Union[torch.Tensor, np.ndarray]] = None
            ):
        super().__init__(metadata, data_key, device)
        self.adj = adj  # Store the adjacency matrix if provided externally

    def _init_transforms(self) -> None:
        super()._init_transforms()
        self._data_transform_ew = make_transform(self.metadata['config'].get('transform_ew', None))
        if self._data_transform_ew.delay > 0:
            msg = "Edge weight transformations with delay embedding are not supported."
            logger.error(msg)
            raise ValueError(msg)
        self._data_transform_ea = make_transform(self.metadata['config'].get('transform_ea', None))
        if self._data_transform_ea.delay > 0:
            msg = "Edge attribute transformations with delay embedding are not supported."
            logger.error(msg)
            raise ValueError(msg)

    # --------------
    # Public interface - for modification
    # --------------
    def set_transforms(
            self,
            metadata: Dict | None = None,
            trajmgr: Optional['TrajectoryManagerGraph'] = None
            ) -> None:
        super().set_transforms(metadata, trajmgr)

        if metadata is not None:
            if "transform_ew_state" in metadata:
                self._data_transform_ew.load_state_dict(metadata["transform_ew_state"])
            if "transform_ea_state" in metadata:
                self._data_transform_ea.load_state_dict(metadata["transform_ea_state"])
        else:
            self._data_transform_ew.load_state_dict(trajmgr._data_transform_ew.state_dict())
            if hasattr(trajmgr, "_data_transform_ea") and trajmgr._data_transform_ea is not None:
                self._data_transform_ea.load_state_dict(trajmgr._data_transform_ea.state_dict())
            if hasattr(trajmgr, "_data_transform_p") and trajmgr._data_transform_p is not None:
                self._data_transform_p.load_state_dict(trajmgr._data_transform_p.state_dict())
        self.metadata["transform_ew_state"] = self._data_transform_ew.state_dict()
        self.metadata["transform_ea_state"] = self._data_transform_ea.state_dict() if self._data_transform_ea is not None else None
        self._transform_fitted = True

    # --------------
    # Public interface - for workflow
    # --------------
    # None
    # Reuse parent class methods

    # --------------
    # Workflow implementation - not meant for public use
    # --------------
    def load_data(self) -> Dict:
        data = super().load_data()

        # By now t/x/y/u/p should have been loaded.
        ei = data.get('ei', None)
        ew = data.get('ew', None)
        ea = data.get('ea', None)

        adj = data.get('adj', None)
        if adj is not None:
            if self.adj is not None:
                logger.warning("Adjacency matrix provided both externally and in data file. Using the one from data.")
            self.adj = adj
            logger.info("Loaded adjacency matrix from data file")

        # Process ei and ew
        if ei is not None:
            if self.adj is not None:
                logger.warning("Edge index provided both externally and in data file. Using the one from data.")
        else:
            logger.info("Edge index is not in data, generating from adjacency matrix")
            ei, ew = adj_to_edge(self.adj)
        self.ei = _process_data(ei, self.x, "ei", base_dim=2, offset=0)
        self.ew = _process_data(ew, self.x, "ew", base_dim=1, offset=0)

        # Process ea
        self.ea = _process_data(ea, self.x, "ea", base_dim=2, offset=0)

        # Count nodes
        _n = []
        for _e in self.ei:
            for _ee in _e:
                _n.append(np.max(_ee) + 1)
        self.n_nodes = int(np.max(_n))
        self.metadata["n_nodes"] = self.n_nodes
        logger.info(f"Number of nodes detected: {self.n_nodes}")

        return data

    def data_truncation(self) -> None:
        super().data_truncation()

        # Update n_state_features, n_aux_features, n_control_features to per-node basis
        assert self.metadata["n_state_features"] % self.n_nodes == 0, \
            "Total number of state features must be divisible by number of nodes."
        assert self.metadata["n_aux_features"] % self.n_nodes == 0, \
            "Total number of auxiliary features must be divisible by number of nodes."
        assert self.metadata["n_control_features"] % self.n_nodes == 0, \
            "Total number of control features must be divisible by number of nodes."
        self.metadata["n_state_features"] = self.metadata["n_state_features"] // self.n_nodes
        self.metadata["n_aux_features"] = self.metadata["n_aux_features"] // self.n_nodes
        self.metadata["n_control_features"] = self.metadata["n_control_features"] // self.n_nodes
        logger.info(f"Number of state features, updated for graph: {self.metadata['n_state_features']}")
        logger.info(f"Number of auxiliary features, updated for graph: {self.metadata['n_aux_features']}")
        logger.info(f"Number of control features, updated for graph: {self.metadata['n_control_features']}")

        # Graph specific truncation for ei, ew, ea
        cfg = self.metadata['config'].get(self.data_key, {})
        n_samples: Optional[int] = cfg.get("n_samples", None)
        n_steps: Optional[int] = cfg.get("n_steps", None)
        # Subset trajectories if n_samples is provided.
        if n_samples is not None:
            if n_samples > 1:
                self.ei = self.ei[:n_samples]
                self.ew = self.ew[:n_samples]
                self.ea = self.ea[:n_samples]

        # Truncate each trajectory's length if n_steps is provided.
        if n_steps is not None:
            self.ei = [_ei[:n_steps] for _ei in self.ei]
            self.ew = [_ew[:n_steps] for _ew in self.ew]
            self.ea = [_ea[:n_steps] for _ea in self.ea]

        # Complete metadata
        self.metadata["n_edge_weights"] = 1 if self.ew[0][0].size > 0 else 0
        self.metadata["n_edge_features"] = int(self.ea[0][0].shape[-1])
        logger.info(f"Number of edge features: {self.metadata['n_edge_features']}")
        logger.info(f"Number of edge weights: {self.metadata['n_edge_weights']}")

    def apply_data_transformations(self) -> None:
        """
        Apply data transformations to the loaded trajectories and control inputs.
        This creates the dataset.

        The raw data is expected to be [T, n_nodes * n_features], but the transformation
        assumes [T * n_nodes, n_features].  So extra reshaping is needed.
        """
        assert self.data_index is not None, "Dataset must be split before applying transformations."

        if not self._transform_fitted:
            logger.info("Fitting transformation for state features.")
            X = [self._graph_data_reshape(self.x[i], forward=True) for i in self.data_index]
            self._data_transform_x.fit(np.vstack(X))  # Make sure the input is 3D
            self.metadata["transform_x_state"] = self._data_transform_x.state_dict()

            if self.metadata["n_aux_features"] > 0:
                logger.info("Fitting transformation for auxiliary features.")
                Y = [self._graph_data_reshape(self.y[i], forward=True) for i in self.data_index]
                self._data_transform_y.fit(np.vstack(Y))
                self.metadata["transform_y_state"] = self._data_transform_y.state_dict()

            if self.metadata["n_control_features"] > 0:
                logger.info("Fitting transformation for control inputs.")
                U = [self._graph_data_reshape(self.u[i], forward=True) for i in self.data_index]
                self._data_transform_u.fit(np.vstack(U))
                self.metadata["transform_u_state"] = self._data_transform_u.state_dict()

            if self.metadata["n_parameters"] > 0:
                logger.info("Fitting transformation for parameters.")
                P = [self._graph_data_reshape(self.p[i][...,None,:], forward=True) for i in self.data_index]
                self._data_transform_p.fit(np.vstack(P))
                self.metadata["transform_p_state"] = self._data_transform_p.state_dict()

            # For edges, stack all edges and fit the transformations.
            if self.metadata["n_edge_weights"] > 0:
                logger.info("Fitting transformation for edge weights.")
                E = [np.hstack(self.ew[i]).reshape(-1,1) for i in self.data_index]
                self._data_transform_ew.fit(E)
                self.metadata["transform_ew_state"] = self._data_transform_ew.state_dict()

            if self.metadata["n_edge_features"] > 0:
                logger.info("Fitting transformation for edge features.")
                E = [np.vstack(self.ea[i]) for i in self.data_index]
                self._data_transform_ea.fit(E)
                self.metadata["transform_ea_state"] = self._data_transform_ea.state_dict()
        else:
            logger.info("Transformations already fitted. Skipping fitting step.")

        logger.info("Applying transformations to state features and control inputs.")
        self.dataset = self._transform_by_index(self.data_index)

        if self.metadata["delay"] > 0:
            logger.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        # Process X first
        # If the common delay is larger (larger delay in u), we trim out the first few steps.
        # This way the latest x and u are aligned.
        tmp = [self._graph_data_reshape(self.x[i], forward=True) for i in indices]
        _X = [np.array(self._data_transform_x.transform(_t)) for _t in tmp]
        _d = self.metadata["delay"] - self._data_transform_x.delay
        if _d > 0:
            _X = [x[:, _d:] for x in _X]

        _T = [self.t[i] for i in indices]
        if self.metadata["delay"] > 0:
            _T = [t[self.metadata["delay"]:] for t in _T]

        if self.metadata["n_aux_features"] > 0:
            tmp = [self._graph_data_reshape(self.y[i], forward=True) for i in indices]
            _Y = [np.array(self._data_transform_y.transform(_t)) for _t in tmp]
            _d = self.metadata["delay"]
            if _d > 0:
                _Y = [y[:, _d:] for y in _Y]
        else:
            _Y = [None for _ in _X]

        if self.metadata["n_control_features"] > 0:
            tmp = [self._graph_data_reshape(self.u[i], forward=True) for i in indices]
            _U = [np.array(self._data_transform_u.transform(_t)) for _t in tmp]
            _d = self.metadata["delay"] - self._data_transform_u.delay
            if _d > 0:
                _U = [u[:, _d:] for u in _U]
        else:
            _U = [None for _ in _X]

        if self.metadata["n_parameters"] > 0:
            tmp = [self._graph_data_reshape(self.p[i][..., None, :], forward=True) for i in indices]
            _P = [self._data_transform_p.transform(_t) for _t in tmp]
        else:
            _P = [None for _ in _X]

        # Graph data is not delayed, just truncate out the initial steps.
        _delay = self.metadata["delay"]
        _Ei = [self.ei[i][_delay:] for i in indices]

        if self.metadata["n_edge_weights"] > 0:
            _Ew = []
            for i in indices:
                _tmp = self._data_transform_ew.transform([e.reshape(-1,1) for e in self.ew[i][_delay:]])
                _Ew.append([_t.reshape(-1) for _t in _tmp])
        else:
            _Ew = [None for _ in _X]

        if self.metadata["n_edge_features"] > 0:
            _Ea = [self._data_transform_ea.transform(self.ea[i][_delay:]) for i in indices]
        else:
            _Ea = [None for _ in _X]

        # Lastly assemble the dataset.
        dataset = []
        for _t, _x, _y, _u, _p, _ei, _ew, _ea in zip(_T, _X, _Y, _U, _P, _Ei, _Ew, _Ea):
            dataset.append(DynData(
                t=torch.tensor(_t, dtype=self.dtype, device=self.device),
                x=torch.tensor(self._graph_data_reshape(_x, forward=False), dtype=self.dtype, device=self.device),
                y=torch.tensor(self._graph_data_reshape(_y, forward=False), dtype=self.dtype, device=self.device) if _y is not None else None,
                u=torch.tensor(self._graph_data_reshape(_u, forward=False), dtype=self.dtype, device=self.device) if _u is not None else None,
                p=torch.tensor(self._graph_data_reshape(_p, forward=False).squeeze(-2), dtype=self.dtype, device=self.device) if _p is not None else None,
                ei=[torch.tensor(_e, dtype=torch.long) for _e in _ei],
                ew=[torch.tensor(_e, dtype=self.dtype) for _e in _ew] if _ew is not None else None,
                ea=[torch.tensor(_a, dtype=self.dtype) for _a in _ea] if _ea is not None else None,
            ))
        return dataset

    def _graph_data_reshape(self, data: np.ndarray, forward: bool) -> np.ndarray:
        """
        Reshape the raw data between [T, n_nodes * n_features] and [n_nodes, T, n_features].

        The 0th axis is as if batch.
        """
        if forward:
            # Reshape from [T, n_nodes * n_features] to [n_nodes, T, n_features]
            tmp = data.reshape(data.shape[0], self.n_nodes, -1)  # [T, n_nodes, n_features_per_node]
            return np.swapaxes(tmp, 0, 1)  # [n_nodes, T, n_features_per_node]

        # Reshape from [n_nodes, T, n_features] to [T, n_nodes * n_features]
        tmp = np.swapaxes(data, 0, 1)  # [T, n_nodes, n_features_per_node]
        return tmp.reshape(tmp.shape[0], -1)

    def create_dataloaders(self) -> None:
        """
        For graph data, we aggregate the trajectories into batches of graphs.
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)
        if_shuffle: bool = dl_cfg.get("shuffle", True)

        n_batch = int(np.ceil(len(self.dataset) / batch_size))
        _data = []
        for i in range(n_batch):
            _data.append(DynData.collate(self.dataset[i*batch_size:(i+1)*batch_size]))

        logger.info(f"Creating dataloaders with batch size 1 for graph data, aggregated from {batch_size} samples.")
        self.dataloader = DataLoader(_data, batch_size=1, shuffle=if_shuffle, collate_fn=DynData.collate)
