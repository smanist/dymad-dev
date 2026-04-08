import copy
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from os import PathLike
from typing import Any

import numpy as np
import torch

from dymad.core.graph_series import GraphSeriesBatch
from dymad.core.model_context import build_model_context
from dymad.core.runtime import (
    RaggedGraphRuntime,
    RaggedRegularRuntime,
    TypedRuntime,
    UniformGraphRuntime,
    UniformRegularRuntime,
)
from dymad.core.series import RegularSeriesBatch
from dymad.core.torch_transforms import AutoencoderTransform, ComposeTransform
from dymad.core.transform_builder import build_transform_module
from dymad.core.transform_module import FieldTransformModule, SeriesTransformPipeline
from dymad.exec.context import ExecutionContext, build_default_context
from dymad.exec.state import PredictionWorkflowPlan
from dymad.io.series_adapter import SeriesAdapter
from dymad.io.trajectory_manager import TrajectoryManager
from dymad.utils.misc import load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundaryLoadTrace:
    plan: PredictionWorkflowPlan
    model_ref: str


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
    tmp = data.reshape(*shp, nnd, -1)  # [..., T, n_nodes, n_states_per_node]
    tmp = np.swapaxes(tmp, -3, -2)  # [..., n_nodes, T, n_states_per_node]  Needed for time delay
    tmp = tmp.reshape(-1, *tmp.shape[-2:])  # [all_nodes, T, n_states_per_node]
    return tmp


def _infer_graph_nodes(edge_index) -> int:
    if isinstance(edge_index, (np.ndarray, torch.Tensor)):
        tensor = torch.as_tensor(edge_index)
        return int(tensor.max().item()) + 1
    if (
        isinstance(edge_index, (list, tuple))
        and edge_index
        and isinstance(edge_index[0], (np.ndarray, torch.Tensor, list, tuple))
    ):
        if isinstance(edge_index[0], (list, tuple)):
            values = [torch.as_tensor(step).max().item() for step in edge_index]
        else:
            values = [torch.as_tensor(step).max().item() for step in edge_index]
        return int(max(values)) + 1
    if (
        isinstance(edge_index, list)
        and edge_index
        and isinstance(edge_index[0], (np.ndarray, torch.Tensor))
    ):
        values = [torch.as_tensor(step).max().item() for step in edge_index]
        return int(max(values)) + 1
    raise ValueError(
        "Typed graph checkpoint prediction currently supports fixed or per-step single-graph edge indices."
    )


def _transform_graph_node_payload(data, nnd, transform_module, dtype, device):
    array = np.asarray(data)
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)

    batch_size = array.shape[0]
    node_major = graph_data_prep(array, nnd)
    transformed = transform_module.transform_batch(
        _to_tensor_batch(node_major, dtype=dtype, device=device)
    )
    stacked = torch.stack(transformed)
    return stacked.reshape(batch_size, nnd, stacked.shape[-2], stacked.shape[-1]).permute(
        0, 2, 1, 3
    )


def _ensure_regular_batch(data):
    array = np.asarray(data)
    if array.ndim == 1:
        return np.expand_dims(array, axis=(0, 1))
    if array.ndim == 2:
        return np.expand_dims(array, axis=0)
    return array


def _ensure_param_batch(data):
    array = np.asarray(data)
    if array.ndim == 0:
        return np.expand_dims(array, axis=(0, 1))
    if array.ndim == 1:
        return np.expand_dims(array, axis=0)
    return array


def _infer_prediction_delay(
    *, raw_state, transformed_state, raw_control=None, transformed_control=None
) -> int:
    raw_state_batch = _ensure_regular_batch(raw_state)
    delays = [int(raw_state_batch.shape[1] - transformed_state.shape[1])]
    if raw_control is not None and transformed_control is not None:
        raw_control_batch = _ensure_regular_batch(raw_control)
        delays.append(int(raw_control_batch.shape[1] - transformed_control.shape[1]))
    valid = [delay for delay in delays if delay > 0]
    return max(valid) if valid else 0


def _trim_graph_temporal_payload(payload, *, delay: int, target_steps: int):
    if payload is None or delay <= 0:
        return payload
    if isinstance(payload, list):
        if len(payload) == target_steps + delay:
            return payload[delay:]
        return payload
    if isinstance(payload, tuple):
        if len(payload) == target_steps + delay:
            return payload[delay:]
        return payload
    if isinstance(payload, torch.Tensor):
        if payload.ndim >= 1 and payload.shape[0] == target_steps + delay:
            return payload[delay:]
        return payload
    array = np.asarray(payload)
    if array.ndim >= 1 and array.shape[0] == target_steps + delay:
        return array[delay:]
    return payload


def _to_tensor_batch(
    data,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[torch.Tensor]:
    array = _ensure_regular_batch(data)
    return [torch.as_tensor(item, dtype=dtype, device=device) for item in array]


def _stack_batch(outputs: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(outputs)
    if len(outputs) == 1:
        return stacked[0]
    return stacked


def _normalize_graph_batch_payload(payload, batch_size: int):
    if payload is None:
        return [None] * batch_size
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        if len(payload) != batch_size:
            raise ValueError("Nested graph payload batch size must match the input batch size.")
        return payload
    return [payload for _ in range(batch_size)]


def _runtime_with_params(runtime: TypedRuntime, params: torch.Tensor | None) -> TypedRuntime:
    if params is None:
        return runtime
    if isinstance(
        runtime,
        (UniformRegularRuntime, RaggedRegularRuntime, UniformGraphRuntime, RaggedGraphRuntime),
    ):
        return replace(runtime, params=params)
    return runtime


def _runtime_prediction_delay(runtime: TypedRuntime) -> int:
    metas = tuple(getattr(runtime, "meta", ()) or ())
    delays = {
        int(meta.get("delay", 0))
        for meta in metas
        if isinstance(meta, dict) and meta.get("delay", 0)
    }
    if not delays:
        return 0
    if len(delays) == 1:
        return next(iter(delays))
    return max(delays)


def _align_prediction_time_input(t, runtime: TypedRuntime):
    delay = _runtime_prediction_delay(runtime)
    if delay <= 0 or t is None:
        return t

    n_steps = getattr(runtime, "n_steps", None)
    if n_steps is None:
        return t

    if isinstance(t, torch.Tensor):
        if t.ndim == 1:
            if t.shape[0] == n_steps + delay:
                return t[delay:]
            return t
        if t.ndim == 2:
            if t.shape[1] == n_steps + delay:
                return t[:, delay:]
            return t
        return t

    array = np.asarray(t)
    if array.ndim == 1:
        if array.shape[0] == n_steps + delay:
            return array[delay:]
        return t
    if array.ndim == 2:
        if array.shape[1] == n_steps + delay:
            return array[:, delay:]
        return t
    return t


def _prediction_defaults_from_config(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}

    phases = config.get("phases")
    if not isinstance(phases, list):
        return {}

    for phase in reversed(phases):
        if not isinstance(phase, dict):
            continue
        if "trainer" not in phase:
            continue
        defaults = {}
        if "ode_method" in phase:
            defaults["method"] = phase["ode_method"]
        defaults.update(copy.deepcopy(phase.get("ode_args", {})))
        if defaults:
            return defaults
    return {}


def _load_model_checkpoint(model_class, checkpoint_path):
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
        chkpt_path = os.path.join(chkpt_path.split(".")[0], chkpt_path)
        if not os.path.exists(chkpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {chkpt_path}.")
    chkpt = torch.load(chkpt_path, weights_only=False)
    cfg = chkpt["config"]
    md = chkpt["train_md"]
    prediction_defaults = _prediction_defaults_from_config(cfg)
    dtype = torch.double if cfg["data"].get("double_precision", False) else torch.float
    torch.set_default_dtype(dtype)  # GNNs use the default dtype, so we need to set it here

    # Model
    model_config = cfg.get("model", None)
    model = model_class(model_config, md, dtype=dtype)
    model.load_state_dict(chkpt["model_state_dict"])

    # Data transformations
    _data_transform_x = build_transform_module(
        cfg.get("transform_x", None), md["transform_x_state"]
    )

    _has_u = md.get("transform_u_state", None) is not None
    if _has_u:
        _data_transform_u = build_transform_module(
            cfg.get("transform_u", None), md["transform_u_state"]
        )

    _has_ew = cfg.get("transform_ew", None) is not None
    if _has_ew:
        _data_transform_ew = build_transform_module(
            cfg.get("transform_ew", None), md["transform_ew_state"]
        )

    _has_ea = cfg.get("transform_ea", None) is not None
    if _has_ea:
        _data_transform_ea = build_transform_module(
            cfg.get("transform_ea", None), md["transform_ea_state"]
        )

    _regular_pipeline = SeriesTransformPipeline(
        [
            FieldTransformModule("state", _data_transform_x),
            *([FieldTransformModule("control", _data_transform_u)] if _has_u else []),
        ]
    )

    # Data processing
    def _proc_x0(x0, device):
        transformed = _stack_batch(
            _data_transform_x.transform_batch(_to_tensor_batch(x0, dtype=dtype, device=device))
        )
        if transformed.ndim == 2:
            return transformed[0]
        return transformed[:, 0, :]

    def _proc_u(us, device):
        return None

    if _has_u:

        def _proc_u(us, device):
            transformed = _stack_batch(
                _data_transform_u.transform_batch(_to_tensor_batch(us, dtype=dtype, device=device))
            )
            if transformed.ndim == 3:
                return transformed[0]
            return transformed

    def _build_regular_prediction_payload(x0, u, p, device):
        x_batch = _ensure_regular_batch(x0)
        u_batch = _ensure_regular_batch(u) if u is not None else None
        p_batch = _ensure_param_batch(p) if p is not None else None
        items = []
        for index in range(x_batch.shape[0]):
            control = None if u_batch is None else u_batch[index]
            params = None if p_batch is None else p_batch[index, 0]
            items.append(
                SeriesAdapter.from_regular_arrays(
                    np.arange(x_batch.shape[1], dtype=np.int64),
                    x_batch[index],
                    control=control,
                    params=params,
                    dtype=dtype,
                    device=device,
                )
            )
        return _regular_pipeline(RegularSeriesBatch.collate(items))

    def _build_graph_prediction_payload(x0, u, p, ei, ew, ea, device):
        array = np.asarray(x0)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)

        batch_size = array.shape[0]
        edge_index_payloads = _normalize_graph_batch_payload(ei, batch_size)
        nnd = _infer_graph_nodes(edge_index_payloads[0])
        node_state = _transform_graph_node_payload(x0, nnd, _data_transform_x, dtype, device)
        control = None
        if _has_u and u is not None:
            control = _transform_graph_node_payload(u, nnd, _data_transform_u, dtype, device)
        p_batch = None if p is None else _ensure_param_batch(p)
        delay = _infer_prediction_delay(
            raw_state=x0,
            transformed_state=node_state,
            raw_control=u,
            transformed_control=control,
        )

        edge_weight_payloads = _normalize_graph_batch_payload(_proc_ew(ew, device), batch_size)
        edge_attr_payloads = _normalize_graph_batch_payload(_proc_ea(ea, device), batch_size)

        items = []
        for index in range(node_state.shape[0]):
            target_steps = int(node_state.shape[1])
            items.append(
                SeriesAdapter.from_graph_arrays(
                    np.arange(target_steps, dtype=np.int64),
                    node_state[index],
                    edge_index=_trim_graph_temporal_payload(
                        edge_index_payloads[index],
                        delay=delay,
                        target_steps=target_steps,
                    ),
                    control=None if control is None else control[index],
                    params=None if p_batch is None else p_batch[index, 0],
                    edge_weight=_trim_graph_temporal_payload(
                        edge_weight_payloads[index],
                        delay=delay,
                        target_steps=target_steps,
                    ),
                    edge_attr=_trim_graph_temporal_payload(
                        edge_attr_payloads[index],
                        delay=delay,
                        target_steps=target_steps,
                    ),
                    dtype=dtype,
                    device=device,
                    meta={"delay": delay} if delay > 0 else None,
                )
            )
        return build_model_context(GraphSeriesBatch.collate(items))

    def _proc_ew(ew, device):
        return ew

    if _has_ew:

        def _proc_ew(ew, device):
            if isinstance(ew, list) and not isinstance(ew[0], list):
                payloads = [
                    torch.as_tensor(_e.reshape(-1, 1), dtype=dtype, device=device) for _e in ew
                ]
                _tmp = _data_transform_ew.transform_batch(payloads)
                return [step.reshape(-1) for step in _tmp]
            elif isinstance(ew[0], list):
                _ew = []
                for e in ew:
                    payloads = [
                        torch.as_tensor(_e.reshape(-1, 1), dtype=dtype, device=device) for _e in e
                    ]
                    _tmp = _data_transform_ew.transform_batch(payloads)
                    _ew.append([step.reshape(-1) for step in _tmp])
            else:
                raise ValueError("Edge weights format not recognized.")
            return _ew

    def _proc_ea(ea, device):
        return ea

    if _has_ea:

        def _proc_ea(ea, device):
            if isinstance(ea, list) and not isinstance(ea[0], list):
                payloads = [torch.as_tensor(_e, dtype=dtype, device=device) for _e in ea]
                _tmp = _data_transform_ea.transform_batch(payloads)
                return _tmp
            elif isinstance(ea[0], list):
                _ea = []
                for e in ea:
                    payloads = [torch.as_tensor(_e, dtype=dtype, device=device) for _e in e]
                    _tmp = _data_transform_ea.transform_batch(payloads)
                    _ea.append(_tmp)
            else:
                raise ValueError("Edge attributes format not recognized.")
            return _ea

    def _proc_prd(pred):
        outputs = _data_transform_x.inverse_batch(_to_tensor_batch(pred, dtype=dtype, device="cpu"))
        result = _stack_batch(outputs).cpu().numpy()
        if result.ndim == 3 and result.shape[0] == 1:
            return result[0]
        return result

    # Prediction in data space
    def predict_fn(
        x0,
        t,
        u=None,
        p=None,
        ei=None,
        ew=None,
        ea=None,
        device="cpu",
        ret_dat=False,
        **predict_kwargs,
    ):
        """Predict trajectory in data space."""
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(device=device)
        _has_graph = ei is not None
        if ei is None:
            regular_payload = _build_regular_prediction_payload(x0, u, p, device)
            regular_context = build_model_context(regular_payload)
            _x0 = regular_context.initial_state_tensor(squeeze_single=True)
            _data = regular_context.to_runtime()
        else:
            graph_context = _build_graph_prediction_payload(x0, u, p, ei, ew, ea, device)
            _x0 = graph_context.initial_state_tensor(squeeze_single=True)
            _data = graph_context.to_runtime()
        _data = _runtime_with_params(
            _data,
            None if p is None else torch.as_tensor(p, dtype=dtype, device=device),
        )
        pred_t = _align_prediction_time_input(t, _data)

        if ret_dat:
            return {
                "t": _data.t,
                "x": _x0,
                "u": _data.u,
                "p": _data.p,
                "ei": getattr(_data, "ei", None),
                "ew": getattr(_data, "ew", None),
                "ea": getattr(_data, "ea", None),
            }

        with torch.no_grad():
            effective_predict_kwargs = dict(prediction_defaults)
            effective_predict_kwargs.update(predict_kwargs)
            pred = model.predict(_x0, _data, pred_t, **effective_predict_kwargs).cpu().numpy()

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
            tmp = np.swapaxes(tmp, -3, -2)  # [all_nodes, T', n_features_per_node]
            shp = tmp.shape[:-2]  # [all_nodes]
            nnd = _data.n_nodes // _data.batch_size  # n_nodes
            shp = (*shp[:-1], _data.batch_size, nnd)  # [batch_size, n_nodes]
            tmp = tmp.reshape(
                -1, *tmp.shape[-2:]
            )  # [:, T', n_features_per_node]  Needed for time delay
            prd = _proc_prd(tmp)  # [:, T, n_states_per_node]  Might change T
            prd = prd.reshape(*shp, *prd.shape[-2:])  # [batch_size, n_nodes, T, n_states_per_node]
            prd = np.swapaxes(prd, -3, -2)  # [batch_size, T, n_nodes, n_states_per_node]
            prd = prd.reshape(*prd.shape[:-2], -1)  # [batch_size, T, n_nodes*n_states_per_node]
            if prd.ndim > x0.ndim:
                prd = prd.squeeze(0)  # Squeeze out the leading dim if exists
            return prd
        return _proc_prd(pred)

    return model, predict_fn


def load_model(
    model_class,
    checkpoint_path: str | PathLike[str],
    *,
    context: ExecutionContext | None = None,
    horizon: int = 1,
    has_control: bool = False,
    has_graph: bool = False,
    return_trace: bool = False,
):
    """Load a model from a checkpoint and optionally record the boundary plan."""
    active_context = context or build_default_context()
    model_module = getattr(model_class, "__module__", type(model_class).__module__)
    model_name = getattr(model_class, "__name__", type(model_class).__name__)
    model_ref = f"{model_module}:{model_name}"
    plan = active_context.executor.plan_checkpoint_prediction(
        model_ref=model_ref,
        checkpoint_path=str(checkpoint_path),
        horizon=horizon,
        has_control=has_control,
        has_graph=has_graph,
    )
    model, predict_fn = _load_model_checkpoint(model_class, str(checkpoint_path))
    if return_trace:
        return (
            model,
            predict_fn,
            BoundaryLoadTrace(
                plan=plan,
                model_ref=model_ref,
            ),
        )
    return model, predict_fn


def _prepare_visualize_model_input(
    input_data: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None],
) -> dict:
    prepared = dict(input_data)

    t = prepared.get("t")
    if isinstance(t, torch.Tensor):
        if t.ndim >= 2:
            prepared["t"] = t[:, :1]
        elif t.ndim == 1 and t.numel() > 1:
            prepared["t"] = t[:1]

    u = prepared.get("u")
    if isinstance(u, torch.Tensor):
        if u.ndim >= 3:
            prepared["u"] = u[:, :1, ...]
        elif u.ndim == 2 and u.shape[0] > 1:
            prepared["u"] = u[:1, ...]

    for key in ("ei", "ew", "ea"):
        payload = prepared.get(key)
        if getattr(payload, "is_nested", False):
            prepared[key] = [item for item in payload.unbind()]

    return prepared


def visualize_model(
    mdl_class=None,
    checkpoint_path=None,
    model=None,
    prd_func=None,
    ref_data=None,
    depth=1,
    device="cpu",
    ifsave=False,
):
    try:
        from torchview import draw_graph
    except ImportError as e:
        raise ImportError(
            "Visualization requires optional dependency 'torchview'.\n"
            "Install via: pip install dymad[viz]"
        ) from e

    if mdl_class is None:
        assert model is not None and prd_func is not None, (
            "Either mdl_class and checkpoint_path, or model and prd_func must be provided."
        )
    else:
        assert checkpoint_path is not None, (
            "checkpoint_path must be provided when mdl_class is given."
        )
        model, prd_func = load_model(mdl_class, checkpoint_path)

    if isinstance(ref_data, str):
        dat = np.load(ref_data, allow_pickle=True)
    else:
        dat = ref_data  # Assuming dict
    t_data = dat.get("t", None)
    x_data = dat.get("x", None)
    u_data = dat.get("u", None)
    p_data = dat.get("p", None)
    ei_data = dat.get("ei", None)
    ew_data = dat.get("ew", None)
    ea_data = dat.get("ea", None)

    input_data = prd_func(
        x_data, t_data, u=u_data, p=p_data, ei=ei_data, ew=ew_data, ea=ea_data, ret_dat=True
    )
    input_data = _prepare_visualize_model_input(input_data)

    model_graph = draw_graph(model, input_data=input_data, depth=depth, device=device)

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

    def __init__(
        self,
        model_class: type[torch.nn.Module] | None = None,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        config_mod: dict | None = None,
        device: torch.device | None = None,
    ):
        self.device = (
            device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        metadata, self.has_model = self._init_metadata(checkpoint_path, config_path, config_mod)
        self._setup_data(metadata)

        if self.has_model:
            self.model, self.prd_func = load_model(model_class, checkpoint_path)

            def encoder(x):
                x_tensor = torch.as_tensor(x, dtype=self.dtype, device=self.device)
                _x_shape = x_tensor.shape[:-1]
                x_tensor = torch.atleast_2d(x_tensor)
                x_flat = x_tensor.reshape(-1, x_tensor.shape[-1])
                payload = build_model_context(
                    RegularSeriesBatch.collate(
                        SeriesAdapter.from_regular_arrays(
                            time=np.array([0], dtype=np.int64),
                            state=sample.unsqueeze(0),
                            dtype=self.dtype,
                            device=self.device,
                        )
                        for sample in x_flat
                    )
                )
                _z = self.model.encoder(payload)
                return _z.reshape(*_x_shape, -1)

            def decoder(z):
                return self.model.decoder(z, None)

            enc = AutoencoderTransform(self.model, encoder, decoder)
            if not isinstance(self._trans_x, ComposeTransform):
                self._trans_x = ComposeTransform([self._trans_x])
            self._trans_x.append(enc)

        self.NT = self._trans_x.NT

    def _init_metadata(self, checkpoint_path, config_path, config_mod) -> tuple[dict, bool]:
        """Initialize metadata from config or checkpoint."""
        if checkpoint_path is not None:
            path = checkpoint_path
            if not os.path.exists(path):
                path = os.path.join(path.split(".")[0], path)
            assert os.path.exists(path), "Checkpoint path does not exist."
            return torch.load(path, weights_only=False), True
        _config = load_config(config_path, config_mod)
        return {"config": _config}, False

    def _setup_data(self, metadata) -> None:
        """Setup data loaders and datasets.

        Striped from TrainerBase.
        """
        if "train_md" in metadata:
            # Previously processed
            cfg = copy.deepcopy(metadata["train_md"])
            cfg["config"]["dataloader"]["shuffle"] = (
                False  # Turn off shuffling to ensure fixed order of samples
            )
            train = TrajectoryManager(cfg, data_key="train", device=self.device)
            self.train_loader, dataset, _ = train.process_all(typed=True)

            cfg = copy.deepcopy(metadata["valid_md"])
            cfg["config"]["dataloader"]["shuffle"] = (
                False  # Turn off shuffling to ensure fixed order of samples
            )
            valid = TrajectoryManager(cfg, data_key="valid", device=self.device)
            self.valid_loader = valid.process_all(typed=True)[0]

            self.t = dataset[0].time.clone().detach()
            tm = train
        else:
            # Simple config
            # Here we just let train and valid be the same
            tm = TrajectoryManager(metadata, data_key="train", device=self.device)
            self.train_loader, _dataset, _ = tm.process_all(typed=True)
            self.valid_loader = self.train_loader

            self.t = _dataset[0].time.clone().detach()

        self.dtype = tm.dtype
        self._trans_x = tm._data_transform_x
        self._trans_u = tm._data_transform_u

    def encode(self, X: np.ndarray, rng: list | None = None) -> np.ndarray:
        """
        Encode new trajectory data to the observer space.
        """
        _Z = self._trans_x.transform([np.atleast_2d(X)], rng)[0]
        return _Z.squeeze()

    def decode(self, X: np.ndarray, rng: list | None = None) -> np.ndarray:
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
            B = batch.state_tensor().cpu().numpy()[..., :-1, :]  # This is already transformed
            B = B.reshape(-1, B.shape[-1])
            end = self.NT - 1 if self.has_model else self.NT
            B = self._trans_x.inverse_transform([B], [0, end])[
                0
            ]  # A hack to get back to the original space
            F.append(fobs(B))
        return np.hstack(F)

    def get_forward_modes(self, ref=None, rng: list | None = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_forward_modes(ref, rng, **kwargs)

    def get_backward_modes(self, ref=None, rng: list | None = None, **kwargs) -> np.ndarray:
        return self._trans_x.get_backward_modes(ref, rng, **kwargs)
