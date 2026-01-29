import copy
import logging
import numpy as np
import os
import torch
from typing import Callable, Dict, List, Optional, Union, Tuple, Type

from dymad.io.data import DynData
from dymad.io.trajectory_manager import TrajectoryManager
from dymad.transform import Autoencoder, make_transform
from dymad.utils.misc import load_config

logger = logging.getLogger(__name__)

def _atleast_3d(x):
    if x.ndim == 2:
        return np.expand_dims(x, axis=0)
    return x

def graph_data_prep(data, nnd):
    # Some hacking to preprocess graph data for data transforms
    #
    # `data` usually come in shape (..., T, n_nodes * n_states_per_node)
    # where ... is the batch size or None
    # We need to reshape to node-wise data (all_nodes, T, n_states_per_node)
    shp = data.shape[:-1]
    tmp = data.reshape(*shp, nnd, -1)          # [..., T, n_nodes, n_states_per_node]
    tmp = np.swapaxes(tmp, -3, -2)             # [..., n_nodes, T, n_states_per_node]  Needed for time delay
    tmp = tmp.reshape(-1, *tmp.shape[-2:])     # [all_nodes, T, n_states_per_node]
    return tmp

def load_model(model_class, checkpoint_path):
    """
    Load a model from a checkpoint file.

    Args:
        model_class (torch.nn.Module): The class of the model to load.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: A tuple containing the model and a prediction function.

        - nn.Module: The loaded model.
        - callable: A function to predict trajectories in data space.
    """
    # If checkpoint_path does not exist, try adding directory prefix based on filename
    chkpt_path = str(checkpoint_path)
    if not os.path.exists(chkpt_path):
        chkpt_path = os.path.join(chkpt_path.split('.')[0], chkpt_path)
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {chkpt_path}.")
    chkpt = torch.load(chkpt_path, weights_only=False)
    cfg = chkpt['config']
    md = chkpt['train_md']
    dtype = torch.double if cfg['data'].get('double_precision', False) else torch.float
    torch.set_default_dtype(dtype)   # GNNs use the default dtype, so we need to set it here

    # Model
    model_config = cfg.get('model', None)
    model = model_class(model_config, md, dtype=dtype)
    model.load_state_dict(chkpt['model_state_dict'])

    # Data transformations
    _data_transform_x = make_transform(cfg.get('transform_x', None))
    _data_transform_x.load_state_dict(md["transform_x_state"])

    _has_u = md.get('transform_u_state', None) is not None
    if _has_u:
        _data_transform_u = make_transform(cfg.get('transform_u', None))
        _data_transform_u.load_state_dict(md["transform_u_state"])

    _has_ew = cfg.get('transform_ew', None) is not None
    if _has_ew:
        _data_transform_ew = make_transform(cfg.get('transform_ew', None))
        _data_transform_ew.load_state_dict(md["transform_ew_state"])

    _has_ea = cfg.get('transform_ea', None) is not None
    if _has_ea:
        _data_transform_ea = make_transform(cfg.get('transform_ea', None))
        _data_transform_ea.load_state_dict(md["transform_ea_state"])

    # Data processing
    def _proc_x0(x0, device):
        _x0 = np.array(_data_transform_x.transform(_atleast_3d(x0)))[:,0,:]
        if len(_x0) == 1:
            _x0 = _x0[0]
        _x0 = torch.tensor(_x0, dtype=dtype, device=device)
        return _x0

    _proc_u = lambda us, device: None
    if _has_u:
        def _proc_u(us, device):
            _u = np.array(_data_transform_u.transform(_atleast_3d(us)))
            if len(_u) == 1:
                _u = _u[0]
            return torch.tensor(_u, dtype=dtype, device=device)

    _proc_ew = lambda ew, device: None
    if _has_ew:
        def _proc_ew(ew, device):
            if isinstance(ew, list) and not isinstance(ew[0], list):
                _tmp = _data_transform_ew.transform([_e.reshape(-1,1) for _e in ew])
                return [torch.tensor(_e.reshape(-1), dtype=dtype, device=device) for _e in _tmp]
            elif isinstance(ew[0], list):
                _ew = []
                for e in ew:
                    _tmp = _data_transform_ew.transform([_e.reshape(-1,1) for _e in e])
                    _ew.append([torch.tensor(_e.reshape(-1), dtype=dtype, device=device) for _e in _tmp])
            else:
                raise ValueError("Edge weights format not recognized.")
            return _ew

    _proc_ea = lambda ea, device: None
    if _has_ea:
        def _proc_ea(ea, device):
            if isinstance(ea, list) and not isinstance(ea[0], list):
                _tmp = _data_transform_ea.transform([_e for _e in ea])
                return [torch.tensor(_e, dtype=dtype, device=device) for _e in _tmp]
            elif isinstance(ea[0], list):
                _ea = []
                for e in ea:
                    _tmp = _data_transform_ea.transform([_e for _e in e])
                    _ea.append([torch.tensor(_e, dtype=dtype, device=device) for _e in _tmp])
            else:
                raise ValueError("Edge attributes format not recognized.")
            return _ea

    def _proc_prd(pred):
        tmp = np.array(_data_transform_x.inverse_transform(_atleast_3d(pred)))
        if tmp.shape[0] == 1:
            return tmp[0]
        return tmp

    # Prediction in data space
    def predict_fn(x0, t, u=None, p=None, ei=None, ew=None, ea=None, device="cpu", ret_dat=False):
        """Predict trajectory in data space."""
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(device=device)
        _has_graph = ei is not None
        if ei is None:
            if u is None:
                _data = DynData()
            else:
                _u  = _proc_u(u, device)
                _data = DynData(u=_u)
            _x0 = _proc_x0(x0, device)
            _data.batch_size = _x0.shape[0] if _x0.ndim == 2 else 1
        else:
            if isinstance(ei, (np.ndarray, torch.Tensor)):
                ei = torch.as_tensor(ei).to(device=device)
                _ei = [ei for _ in range(t.shape[-1])]
            elif isinstance(ei, list):
                if isinstance(ei[0], (np.ndarray, torch.Tensor)):
                    _ei = [torch.as_tensor(e).to(device=device) for e in ei]
                elif isinstance(ei[0], list):
                    _ei = []
                    for e in ei:
                        _ei.append([torch.as_tensor(_e).to(device=device) for _e in e])
                else:
                    raise ValueError("Edge index format not recognized.")
            else:
                raise ValueError("Edge index format not recognized.")
            _u  = _proc_u(u, device)
            _ew = _proc_ew(ew, device)
            _ea = _proc_ea(ea, device)
            _data = DynData(u=_u, ei=_ei, ew=_ew, ea=_ea)

            # Some hacking to handle graph data
            # `x0` usually come in shape (..., T, n_nodes * n_states_per_node)
            nnd = _data.n_nodes // _data.batch_size
            tmp = graph_data_prep(x0, nnd)             # [all_nodes, T, n_states_per_node]
            _x0 = _proc_x0(tmp, device)                # [all_nodes, n_features_per_node]  Only the first step taken
            _x0 = _x0.reshape(_data.batch_size, -1)    # [batch_size, n_nodes*n_features_per_node]

            # _data.batch_size is tracked in DynData for the graph, so no need to set here

        if p is not None:
            _data.p = torch.as_tensor(p, dtype=dtype, device=device)

        if ret_dat:
            return {
                't': t,
                'x': _x0,
                'u': _data.u,
                'p': _data.p,
                'ei': _data.ei,
                'ew': _data.ew,
                'ea': _data.ea
            }

        with torch.no_grad():
            pred = model.predict(_x0, _data, t).cpu().numpy()

        if _has_graph:
            # Some hacking to handle graph data
            # `pred` always comes in shape (..., T', all_nodes * n_features_per_node)
            # where all_nodes = batch_size * n_nodes, and ... is 1 or None
            # Note there all_nodes=`_data.n_nodes` and batch_size=`_data.batch_size`
            # Using T', as it can be different from the final T (e.g., due to time delay).
            #
            # We first need to reshape to node-wise data (..., T', n_features_per_node)
            # for data transformation to get (..., T, n_states_per_node)
            # Then split all_nodes in batches to get final shape (batch_size, T, n_nodes * n_states_per_node)
            if pred.shape[0] == 1:
                pred = pred[0]  # Squeeze out the leading dim if exists
            # Now pred is of shape (T', all_nodes*n_features_per_node)
            shp = pred.shape[:-1]
            tmp = pred.reshape(*shp, _data.n_nodes, -1)  # [T', all_nodes, n_features_per_node]
            tmp = np.swapaxes(tmp, -3, -2)               # [all_nodes, T', n_features_per_node]
            shp = tmp.shape[:-2]                         # [all_nodes]
            nnd = _data.n_nodes // _data.batch_size      # n_nodes
            shp = (*shp[:-1], _data.batch_size, nnd)     # [batch_size, n_nodes]
            tmp = tmp.reshape(-1, *tmp.shape[-2:])       # [:, T', n_features_per_node]  Needed for time delay
            prd = _proc_prd(tmp)                         # [:, T, n_states_per_node]  Might change T
            prd = prd.reshape(*shp, *prd.shape[-2:])     # [batch_size, n_nodes, T, n_states_per_node]
            prd = np.swapaxes(prd, -3, -2)               # [batch_size, T, n_nodes, n_states_per_node]
            prd = prd.reshape(*prd.shape[:-2], -1)       # [batch_size, T, n_nodes*n_states_per_node]
            if prd.ndim > x0.ndim:
                prd = prd.squeeze(0)  # Squeeze out the leading dim if exists
            return prd
        return _proc_prd(pred)

    return model, predict_fn


def visualize_model(
        mdl_class=None, checkpoint_path=None, model=None, prd_func=None,
        ref_data=None, depth=1, device='cpu', ifsave=False):
    try:
        from torchview import draw_graph
    except ImportError as e:
        raise ImportError(
            "Visualization requires optional dependency 'torchview'.\n"
            "Install via: pip install dymad[viz]"
        ) from e

    if mdl_class is None:
        assert model is not None and prd_func is not None, \
            "Either mdl_class and checkpoint_path, or model and prd_func must be provided."
    else:
        assert checkpoint_path is not None, \
            "checkpoint_path must be provided when mdl_class is given."
        model, prd_func = load_model(mdl_class, checkpoint_path)

    if isinstance(ref_data, str):
        dat = np.load(ref_data, allow_pickle=True)
    else:
        dat = ref_data  # Assuming dict
    t_data = dat.get('t', None)
    x_data = dat.get('x', None)
    u_data = dat.get('u', None)
    p_data = dat.get('p', None)
    ei_data = dat.get('ei', None)
    ew_data = dat.get('ew', None)
    ea_data = dat.get('ea', None)

    input_data = prd_func(
        x_data, t_data, u=u_data, p=p_data, ei=ei_data, ew=ew_data, ea=ea_data,
        ret_dat=True)
    for _k in ['ei', 'ew', 'ea']:
        # Decompose nested tensors so that torchview can handle them
        if input_data[_k] is not None:
            input_data[_k] = (input_data[_k].values(), input_data[_k].offsets())

    model_graph = draw_graph(
        model,
        input_data=input_data,
        depth=depth,
        device=device)

    if ifsave:
        if checkpoint_path is None:
            filename = "model" if isinstance(ifsave, bool) else str(ifsave)
            model_graph.visual_graph.render(f"{filename}.viz", format="png")
        else:
            filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
            model_graph.visual_graph.render(f"{filename}/{filename}.viz", format="png")

    return model_graph.visual_graph


class DataInterface:
    """
    Interface for data transforms, possibly with learned autoencoders.

    It loads the model (if available) and data, sets up the necessary transformations,
    and provides methods to encode, decode, and apply observables.

    Cases:

        - [Priority] checkpoint_path is given: Load the data transforms and model from the checkpoint.
          May contain autoencoders.
        - [Secondary] config_path and/or config_mod is given: Instantiate the data transforms from the config.
          No model (i.e., autoencoders) in this case.
    """
    def __init__(self,
                 model_class: Union[Type[torch.nn.Module], None] = None,
                 checkpoint_path: Union[str, None] = None,
                 config_path: Union[str, None] = None,
                 config_mod: Optional[dict] = None,
                 device: Optional[torch.device] = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metadata, self.has_model = self._init_metadata(checkpoint_path, config_path, config_mod)
        self._setup_data(metadata)

        if self.has_model:
            self.model, self.prd_func = load_model(model_class, checkpoint_path)
            def encoder(x):
                _x_shape = x.shape[:-1]
                _z = self.model.encoder(DynData(x=torch.atleast_2d(torch.as_tensor(x))))
                return _z.reshape(*_x_shape, -1)
            def decoder(z):
                return self.model.decoder(z, None)
            enc = Autoencoder(self.model, encoder, decoder)
            self._trans_x.append(enc)

        self.NT = self._trans_x.NT

    def _init_metadata(self, checkpoint_path, config_path, config_mod) -> Tuple[Dict, bool]:
        """Initialize metadata from config or checkpoint."""
        if checkpoint_path is not None:
            path = checkpoint_path
            if not os.path.exists(path):
                path = os.path.join(path.split('.')[0], path)
            assert os.path.exists(path), "Checkpoint path does not exist."
            return torch.load(path, weights_only=False), True
        _config = load_config(config_path, config_mod)
        return {'config': _config}, False

    def _setup_data(self, metadata) -> None:
        """Setup data loaders and datasets.

        Striped from TrainerBase.
        """
        if 'train_md' in metadata:
            # Previously processed
            cfg = copy.deepcopy(metadata['train_md'])
            cfg['config']['dataloader']['shuffle'] = False   # Turn off shuffling to ensure fixed order of samples
            train = TrajectoryManager(cfg, data_key='train', device=self.device)
            self.train_loader, dataset, _ = train.process_all()

            cfg = copy.deepcopy(metadata['valid_md'])
            cfg['config']['dataloader']['shuffle'] = False   # Turn off shuffling to ensure fixed order of samples
            valid = TrajectoryManager(cfg, data_key='valid', device=self.device)
            self.valid_loader = valid.process_all()[0]

            self.t = dataset[0].t[0].clone().detach()
            tm = train
        else:
            # Simple config
            # Here we just let train and valid be the same
            tm = TrajectoryManager(metadata, data_key='train', device=self.device)
            _dataloader, _dataset, _ = tm.process_all()
            # Turn off shuffling to ensure fixed order of samples
            self.train_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_dataloader.batch_size, shuffle=False, collate_fn=DynData.collate)
            self.valid_loader = self.train_loader

            self.t = _dataset[0].t[0].clone().detach()

        self.dtype = tm.dtype
        self._trans_x = tm._data_transform_x
        self._trans_u = tm._data_transform_u

    def encode(self, X: np.ndarray, rng: Optional[List | None] = None) -> np.ndarray:
        """
        Encode new trajectory data to the observer space.
        """
        _Z = self._trans_x.transform([np.atleast_2d(X)], rng)[0]
        return _Z.squeeze()

    def decode(self, X: np.ndarray, rng: Optional[List | None] = None) -> np.ndarray:
        """
        Decode trajectory data from the observer space.
        """
        _Z = self._trans_x.inverse_transform([np.atleast_2d(X)], rng)[0]
        return _Z.squeeze()

    def apply_obs(self, fobs: Callable) -> np.ndarray:
        """
        Apply a generic observable to the raw data.

        Args:
            fobs (Callable): Observable function. It should accept a 2D array input with each row as one step.
                             The output should be a 1D array, whose ith entry corresponds to the ith step.
        """
        F = []
        for batch in self.train_loader:
            B = batch.x.cpu().numpy()[..., :-1, :]        # This is already transformed
            B = B.reshape(-1, B.shape[-1])
            end = self.NT-1 if self.has_model else self.NT
            B = self._trans_x.inverse_transform([B], [0, end])[0]   # A hack to get back to the original space
            F.append(fobs(B))
        return np.hstack(F)

    def get_forward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_forward_modes(ref, rng, **kwargs)

    def get_backward_modes(self, ref=None, rng: Union[List, None] = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_backward_modes(ref, rng, **kwargs)
