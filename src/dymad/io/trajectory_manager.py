import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional, Union, Tuple, Dict, List

from dymad.core.graph_series import GraphSeries, GraphSeriesBatch
from dymad.core.series import RegularSeries, RegularSeriesBatch
from dymad.core.trainer_batch import GraphTrainerBatch, RegularTrainerBatch
from dymad.core.transform_builder import build_legacy_transform, build_transform_module, export_transform_state
from dymad.core.transform_module import FieldTransformModule, LegacyTransformModuleAdapter, SeriesTransformPipeline
from dymad.io.data import DynData
from dymad.io.series_adapter import DynDataAdapter, SeriesAdapter
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
        self.typed_dataset: list[RegularSeries] | list[GraphSeries] | None = None

        self._init_transforms()
        self._load_metadata(self.metadata, data_key)

    def _init_transforms(self) -> None:
        self._transform_fitted = False
        self._data_transform_x = build_legacy_transform(self.metadata['config'].get('transform_x', None))
        self._data_transform_y = build_legacy_transform(self.metadata['config'].get('transform_y', None))
        self._data_transform_p = build_legacy_transform(self.metadata['config'].get('transform_p', None))
        cfg_transform_u = self.metadata['config'].get('transform_u', None)
        self._data_transform_u = build_legacy_transform(cfg_transform_u)
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
            if metadata.get("transform_y_state") is not None:
                self._data_transform_y.load_state_dict(metadata["transform_y_state"])
            if metadata.get("transform_u_state") is not None:
                self._data_transform_u.load_state_dict(metadata["transform_u_state"])
            if metadata.get("transform_p_state") is not None:
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

    def process_data(
        self,
        *,
        typed: bool = False,
    ) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Latter half of process_all
        """
        self.apply_data_transformations()
        self.create_dataloaders(typed=typed)

        dataset = self.typed_dataset if typed else self.dataset
        logger.info(f"Data processing complete. Data size: {len(dataset)}.")
        return self.dataloader, dataset, self.metadata

    def process_all(
        self,
        *,
        typed: bool = False,
    ) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        """
        Returns:
            A tuple containing: dataloader, dataset, metadata
        """
        self.prepare_data()
        if self.data_index is None:
            self.set_data_index()
        res = self.process_data(typed=typed)
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
        self.typed_dataset = self._transform_regular_series_by_index(self.data_index)
        self.dataset = [DynDataAdapter.from_regular_series(series) for series in self.typed_dataset]

        if self.metadata["delay"] > 0:
            logger.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        series_dataset = self._transform_regular_series_by_index(indices)
        return [DynDataAdapter.from_regular_series(series) for series in series_dataset]

    def create_regular_series_dataset(self, indices: torch.Tensor | List[int] | None = None) -> List[RegularSeries]:
        """Expose the first typed data seam for regular trajectory preprocessing."""
        if indices is None:
            if self.data_index is None:
                raise ValueError("data_index must be set before creating a regular-series dataset")
            indices = self.data_index
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        if getattr(self, "dataset", None) is not None and self.data_index is not None:
            position_by_index = {
                int(raw_index): position
                for position, raw_index in enumerate(self.data_index.tolist())
            }
            if all(int(index) in position_by_index for index in indices.tolist()):
                return [
                    SeriesAdapter.from_dyndata(self.dataset[position_by_index[int(index)]])
                    for index in indices
                ]
        return self._transform_regular_series_by_index(indices)

    def _create_raw_regular_series_by_index(self, indices: torch.Tensor) -> List[RegularSeries]:
        dataset = []
        for index in indices:
            target = self.y[index] if self.metadata["n_aux_features"] > 0 else None
            control = self.u[index] if self.metadata["n_control_features"] > 0 else None
            params = self.p[index] if self.metadata["n_parameters"] > 0 else None
            dataset.append(
                SeriesAdapter.from_regular_arrays(
                    self.t[index],
                    self.x[index],
                    target=target,
                    control=control,
                    params=params,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return dataset

    def _build_regular_transform_pipeline(self) -> SeriesTransformPipeline:
        cfg = self.metadata["config"]
        return SeriesTransformPipeline([
            FieldTransformModule(
                "state",
                build_transform_module(cfg.get("transform_x", None), self.metadata.get("transform_x_state")),
            ),
            *(
                [FieldTransformModule(
                    "control",
                    build_transform_module(cfg.get("transform_u", None), self.metadata.get("transform_u_state")),
                )]
                if self.metadata["n_control_features"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "target",
                    build_transform_module(cfg.get("transform_y", None), self.metadata.get("transform_y_state")),
                )]
                if self.metadata["n_aux_features"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "params",
                    build_transform_module(cfg.get("transform_p", None), self.metadata.get("transform_p_state")),
                    time_varying=False,
                )]
                if self.metadata["n_parameters"] > 0 else []
            ),
        ])

    def _transform_regular_series_by_index(self, indices: torch.Tensor) -> List[RegularSeries]:
        raw_dataset = self._create_raw_regular_series_by_index(indices)
        pipeline = self._build_regular_transform_pipeline()
        transformed = pipeline(RegularSeriesBatch.collate(raw_dataset))
        return list(transformed)

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

    def create_dataloaders(self, *, typed: bool = False) -> None:
        """
        Create dataloaders for the data set.
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)
        if_shuffle: bool = dl_cfg.get("shuffle", True)

        logger.info(f"Creating dataloaders for model with batch size {batch_size}.")
        if typed:
            if self.typed_dataset is None:
                raise ValueError("typed_dataset is not available; apply_data_transformations must run first")
            self.dataloader = DataLoader(
                self.typed_dataset,
                batch_size=batch_size,
                shuffle=if_shuffle,
                collate_fn=RegularTrainerBatch.collate_series,
            )
            return

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
        self._data_transform_ew = build_legacy_transform(self.metadata['config'].get('transform_ew', None))
        if self._data_transform_ew.delay > 0:
            msg = "Edge weight transformations with delay embedding are not supported."
            logger.error(msg)
            raise ValueError(msg)
        self._data_transform_ea = build_legacy_transform(self.metadata['config'].get('transform_ea', None))
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
            if metadata.get("transform_ew_state") is not None:
                self._data_transform_ew.load_state_dict(metadata["transform_ew_state"])
            if metadata.get("transform_ea_state") is not None:
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

        raw_batch = GraphSeriesBatch.collate(self._create_raw_graph_series_by_index(self.data_index))
        pipeline = self._build_graph_transform_pipeline()

        if not self._transform_fitted:
            logger.info("Fitting graph transform pipeline on typed graph series.")
            pipeline.fit(raw_batch)
            self._sync_legacy_graph_transforms_from_pipeline(pipeline)
        else:
            logger.info("Transformations already fitted. Skipping fitting step.")

        logger.info("Applying graph transformations through the typed series pipeline.")
        transformed = list(pipeline(raw_batch))
        self.typed_dataset = transformed
        self.dataset = [DynDataAdapter.from_graph_series(series) for series in transformed]

        if self.metadata["delay"] > 0:
            logger.info("Conforming the time data due to delay.")
            # For time, we remove the last "delay" time steps.
            self.t = [ti[:-self.metadata["delay"]] for ti in self.t]

        self._update_dataset_metadata()

    def _transform_by_index(self, indices: torch.Tensor) -> List[DynData]:
        graph_dataset = self._transform_graph_series_by_index(indices)
        return [DynDataAdapter.from_graph_series(series) for series in graph_dataset]

    def create_graph_series_dataset(self, indices: torch.Tensor | List[int] | None = None) -> List[GraphSeries]:
        """Expose the typed graph-series seam for graph trajectory preprocessing."""
        if indices is None:
            if self.data_index is None:
                raise ValueError("data_index must be set before creating a graph-series dataset")
            indices = self.data_index
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        if getattr(self, "dataset", None) is not None and self.data_index is not None:
            position_by_index = {
                int(raw_index): position
                for position, raw_index in enumerate(self.data_index.tolist())
            }
            if all(int(index) in position_by_index for index in indices.tolist()):
                return [
                    SeriesAdapter.from_dyndata(self.dataset[position_by_index[int(index)]])
                    for index in indices
                ]
        return self._transform_graph_series_by_index(indices)

    def _transform_graph_series_by_index(self, indices: torch.Tensor) -> List[GraphSeries]:
        raw_batch = GraphSeriesBatch.collate(self._create_raw_graph_series_by_index(indices))
        transformed = self._build_graph_transform_pipeline()(raw_batch)
        return list(transformed)

    def _create_raw_graph_series_by_index(self, indices: torch.Tensor) -> List[GraphSeries]:
        dataset = []
        for index in indices:
            target = np.swapaxes(self._graph_data_reshape(self.y[index], forward=True), 0, 1) if self.metadata["n_aux_features"] > 0 else None
            control = np.swapaxes(self._graph_data_reshape(self.u[index], forward=True), 0, 1) if self.metadata["n_control_features"] > 0 else None
            params = self.p[index].reshape(self.n_nodes, -1) if self.metadata["n_parameters"] > 0 else None
            edge_weight = np.stack(self.ew[index], axis=0) if self.metadata["n_edge_weights"] > 0 else None
            edge_attr = np.stack(self.ea[index], axis=0) if self.metadata["n_edge_features"] > 0 else None
            dataset.append(
                SeriesAdapter.from_graph_arrays(
                    time=self.t[index],
                    node_state=np.swapaxes(self._graph_data_reshape(self.x[index], forward=True), 0, 1),
                    control=control,
                    target=target,
                    params=params,
                    edge_index=self.ei[index],
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        return dataset

    def _build_graph_transform_pipeline(self) -> SeriesTransformPipeline:
        cfg = self.metadata["config"]
        return SeriesTransformPipeline([
            FieldTransformModule(
                "node_state",
                build_transform_module(cfg.get("transform_x", None), self.metadata.get("transform_x_state")),
            ),
            *(
                [FieldTransformModule(
                    "control",
                    build_transform_module(cfg.get("transform_u", None), self.metadata.get("transform_u_state")),
                )]
                if self.metadata["n_control_features"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "target",
                    build_transform_module(cfg.get("transform_y", None), self.metadata.get("transform_y_state")),
                )]
                if self.metadata["n_aux_features"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "params",
                    build_transform_module(cfg.get("transform_p", None), self.metadata.get("transform_p_state")),
                    time_varying=False,
                )]
                if self.metadata["n_parameters"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "edge_weight",
                    self._build_graph_edge_transform_module(
                        cfg.get("transform_ew", None),
                        self.metadata.get("transform_ew_state"),
                        field="edge_weight",
                    ),
                )]
                if self.metadata["n_edge_weights"] > 0 else []
            ),
            *(
                [FieldTransformModule(
                    "edge_attr",
                    self._build_graph_edge_transform_module(
                        cfg.get("transform_ea", None),
                        self.metadata.get("transform_ea_state"),
                        field="edge_attr",
                    ),
                )]
                if self.metadata["n_edge_features"] > 0 else []
            ),
        ])

    def _build_graph_edge_transform_module(self, config, state_dict, *, field: str):
        legacy_transform = build_legacy_transform(config)
        if state_dict is not None:
            legacy_transform.load_state_dict(state_dict)
        if field == "edge_weight":
            return LegacyTransformModuleAdapter(
                legacy_transform,
                to_legacy=self._graph_edge_weight_to_legacy,
                from_legacy=self._graph_edge_weight_from_legacy,
                invertibility="approximate",
                supports_gradients="false",
            )
        if field == "edge_attr":
            return LegacyTransformModuleAdapter(
                legacy_transform,
                to_legacy=self._graph_edge_attr_to_legacy,
                from_legacy=self._graph_edge_attr_from_legacy,
                invertibility="approximate",
                supports_gradients="false",
            )
        raise ValueError(f"Unsupported graph edge field: {field}")

    @staticmethod
    def _graph_edge_weight_to_legacy(data: torch.Tensor) -> list[np.ndarray]:
        array = data.detach().cpu().numpy()
        return [step.reshape(-1, 1) for step in array]

    @staticmethod
    def _graph_edge_weight_from_legacy(data, *, reference: torch.Tensor) -> torch.Tensor:
        stacked = np.stack([step.reshape(-1) for step in data], axis=0)
        return torch.as_tensor(stacked, dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _graph_edge_attr_to_legacy(data: torch.Tensor) -> list[np.ndarray]:
        array = data.detach().cpu().numpy()
        return [step for step in array]

    @staticmethod
    def _graph_edge_attr_from_legacy(data, *, reference: torch.Tensor) -> torch.Tensor:
        stacked = np.stack(data, axis=0)
        return torch.as_tensor(stacked, dtype=reference.dtype, device=reference.device)

    def _sync_legacy_graph_transforms_from_pipeline(self, pipeline: SeriesTransformPipeline) -> None:
        stage_by_field = {stage.field: stage.transform for stage in pipeline.stages}

        def _sync(field, legacy_attr, metadata_key):
            module = stage_by_field.get(field)
            if module is None:
                self.metadata[metadata_key] = None
                return
            state = export_transform_state(module)
            getattr(self, legacy_attr).load_state_dict(state)
            self.metadata[metadata_key] = state

        _sync("node_state", "_data_transform_x", "transform_x_state")
        if self.metadata["n_aux_features"] > 0:
            _sync("target", "_data_transform_y", "transform_y_state")
        else:
            self.metadata["transform_y_state"] = None
        if self.metadata["n_control_features"] > 0:
            _sync("control", "_data_transform_u", "transform_u_state")
        else:
            self.metadata["transform_u_state"] = None
        if self.metadata["n_parameters"] > 0:
            _sync("params", "_data_transform_p", "transform_p_state")
        else:
            self.metadata["transform_p_state"] = None
        if self.metadata["n_edge_weights"] > 0:
            _sync("edge_weight", "_data_transform_ew", "transform_ew_state")
        else:
            self.metadata["transform_ew_state"] = None
        if self.metadata["n_edge_features"] > 0:
            _sync("edge_attr", "_data_transform_ea", "transform_ea_state")
        else:
            self.metadata["transform_ea_state"] = None

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

    def create_dataloaders(self, *, typed: bool = False) -> None:
        """
        For graph data, we aggregate the trajectories into batches of graphs.
        """
        dl_cfg = self.metadata['config'].get("dataloader", {})
        batch_size: int = dl_cfg.get("batch_size", 1)
        if_shuffle: bool = dl_cfg.get("shuffle", True)

        if typed:
            if self.typed_dataset is None:
                raise ValueError("typed_dataset is not available; apply_data_transformations must run first")
            logger.info(f"Creating typed graph dataloaders with batch size {batch_size}.")
            self.dataloader = DataLoader(
                self.typed_dataset,
                batch_size=batch_size,
                shuffle=if_shuffle,
                collate_fn=GraphTrainerBatch.collate_series,
            )
            return

        n_batch = int(np.ceil(len(self.dataset) / batch_size))
        _data = []
        for i in range(n_batch):
            _data.append(DynData.collate(self.dataset[i*batch_size:(i+1)*batch_size]))

        logger.info(f"Creating dataloaders with batch size 1 for graph data, aggregated from {batch_size} samples.")
        self.dataloader = DataLoader(_data, batch_size=1, shuffle=if_shuffle, collate_fn=DynData.collate)
