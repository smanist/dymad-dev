import logging
import numpy as np
import os
import torch

from dymad.data import DynData, DynGeoData, make_transform
from dymad.utils.misc import load_config

logger = logging.getLogger(__name__)

def load_checkpoint(model, optimizer, schedulers, ref_checkpoint_path, load_from_checkpoint=False, inference_mode=False):
    """
    Load a checkpoint from the specified path.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): The schedulers to load the state into.
        ref_checkpoint_path (str): Reference path to the checkpoint file - Same as the current case.
        load_from_checkpoint (bool or str): If True, load from ref_checkpoint_path; if str, use it as the path; otherwise, skip loading.
        inference_mode (bool, optional): If True, skip loading optimizer and schedulers.

    Returns:
        tuple: A tuple containing:

        - int: The epoch number from which to continue training.
        - float: The best loss recorded in the checkpoint.
        - list: History of losses.
        - list: History of RMSE of trajectories - can be different from loss.
        - dict: Metadata about the data.
    """
    mode = "Inference" if inference_mode else "Training"
    logger.info(f"{mode} mode is enabled.")

    checkpoint_path = None
    if isinstance(load_from_checkpoint, str):
        checkpoint_path = load_from_checkpoint
    elif load_from_checkpoint:
        checkpoint_path = ref_checkpoint_path

    if checkpoint_path is None:
        logger.info(f"Got load_from_checkpoint={load_from_checkpoint}, resulting in checkpoint_path=None. Starting from scratch.")
        return 0, float("inf"), [], [], None

    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, float("inf"), [], [], None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not inference_mode:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        assert len(schedulers) == len(checkpoint["scheduler_state_dict"]), \
            f"Expected {len(schedulers)} schedulers, but got {len(checkpoint['scheduler_state_dict'])} in checkpoint."
        for i in range(len(schedulers)):
            schedulers[i].load_state_dict(checkpoint["scheduler_state_dict"][i])

        # In this case, we do a new training, so we reset the best loss
        return checkpoint["epoch"], float("inf"), checkpoint["hist"], checkpoint["rmse"], checkpoint["metadata"]

    return checkpoint["epoch"], checkpoint["best_loss"], checkpoint["hist"], checkpoint["rmse"], checkpoint["metadata"]

def save_checkpoint(model, optimizer, schedulers, epoch, best_loss, hist, rmse, metadata, checkpoint_path):
    """
    Save the model, optimizer, and scheduler states to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        schedulers (list[torch.optim.lr_scheduler._LRScheduler]): The schedulers to save.
        epoch (int): The current epoch number.
        best_loss (float): The best loss recorded so far.
        hist (list): The history of losses.
        rmse (list): The history of RMSE of trajectories - can be different from loss.
        metadata (dict): Metadata about the data.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    # logger.info(f"Saving checkpoint to {checkpoint_path}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": [scheduler.state_dict() for scheduler in schedulers],
        "best_loss": best_loss,
        "hist": hist,
        "rmse": rmse,
        "metadata": metadata,
    }, checkpoint_path)

def load_model(model_class, checkpoint_path, config_path, config_mod=None):
    """
    Load a model from a checkpoint file.

    Args:
        model_class (torch.nn.Module): The class of the model to load.
        checkpoint_path (str): Path to the checkpoint file.
        config_path (str): Path to the configuration file.
        config_mod (dict, optional): Dictionary to merge into the config.

    Returns:
        tuple: A tuple containing the model and a prediction function.

        - nn.Module: The loaded model.
        - callable: A function to predict trajectories in data space.
    """
    config = load_config(config_path, config_mod)
    chkpt = torch.load(checkpoint_path, weights_only=False)
    md = chkpt['metadata']

    # Model
    model = model_class(config['model'], md)
    model.load_state_dict(chkpt['model_state_dict'])
    dtype = next(model.parameters()).dtype

    # Check if autonomous
    _is_autonomous = md.get('transform_u_state', None) is None

    # Data transformations
    _data_transform_x = make_transform(md['config'].get('transform_x', None))
    _data_transform_x.load_state_dict(md["transform_x_state"])

    if not _is_autonomous:
        _data_transform_u = make_transform(md['config'].get('transform_u', None))
        _data_transform_u.load_state_dict(md["transform_u_state"])

    # Prediction in data space
    if model.GRAPH:
        if _is_autonomous:
            def predict_fn(x0, t, ei=None, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _data_transform_x.transform([x0])[0][0]
                _x0 = torch.tensor(_x0, dtype=dtype, device=device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynGeoData(None, None, ei), t).cpu().numpy()
                return _data_transform_x.inverse_transform([pred])[0]
        else:
            def predict_fn(x0, us, t, ei=None, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _data_transform_x.transform([x0])[0][0]
                _x0 = torch.tensor(_x0, dtype=dtype, device=device)
                _u  = _data_transform_u.transform([us])[0]
                if isinstance(_u, np.ndarray):
                    _u = torch.tensor(_u, dtype=dtype, device=device)
                else:
                    _u = _u.clone().detach().to(device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynGeoData(None, _u, ei), t).cpu().numpy()
                return _data_transform_x.inverse_transform([pred])[0]
    else:
        if _is_autonomous:
            def predict_fn(x0, t, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _data_transform_x.transform([x0])[0][0]
                _x0 = torch.tensor(_x0, dtype=dtype, device=device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(None, None), t).cpu().numpy()
                return _data_transform_x.inverse_transform([pred])[0]
        else:
            def predict_fn(x0, us, t, device="cpu"):
                """Predict trajectory in data space."""
                _x0 = _data_transform_x.transform([x0])[0][0]
                _x0 = torch.tensor(_x0, dtype=dtype, device=device)
                _u  = _data_transform_u.transform([us])[0]
                if isinstance(_u, np.ndarray):
                    _u = torch.tensor(_u, dtype=dtype, device=device)
                else:
                    _u = _u.clone().detach().to(device)
                with torch.no_grad():
                    pred = model.predict(_x0, DynData(None, _u), t).cpu().numpy()
                return _data_transform_x.inverse_transform([pred])[0]

    return model, predict_fn
