import logging
import os
import torch
import yaml

from keystone.src.data import make_transform

logging=logging.getLogger(__name__)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, load_from_checkpoint, inference_mode=False):
    """
    Load a checkpoint from the specified path.

    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        scheduler: The scheduler to load the state into.
        checkpoint_path (str): Path to the checkpoint file.
        inference_mode (bool, optional): If True, skip loading optimizer and scheduler.

    Returns:
        int: The epoch number from which to continue training.
        float: The best loss recorded in the checkpoint.
    """
    mode = "Inference" if inference_mode else "Training"
    logging.info(f"{mode} mode is enabled.")

    if not os.path.exists(checkpoint_path) or not load_from_checkpoint:
        logging.info(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, float("inf"), [], [], None

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not inference_mode:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["best_loss"], checkpoint["hist"], checkpoint["rmse"], checkpoint["metadata"]

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, hist, rmse, metadata, checkpoint_path):
    """
    Save the model, optimizer, and scheduler states to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The scheduler to save.
        epoch (int): The current epoch number.
        best_loss (float): The best loss recorded so far.
        hist (list): The history of losses.
        rmse (list): The history of RMSE of trajectories - can be different from loss.
        metadata (dict): Metadata about the data.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "hist": hist,
        "rmse": rmse,
        "metadata": metadata,
    }, checkpoint_path)

def load_model(model_class, checkpoint_path, config_path):
    """ Load a model from a checkpoint file.

    Args:
        model_class: The class of the model to load.
        checkpoint_path (str): Path to the checkpoint file.
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    chkpt = torch.load(checkpoint_path, weights_only=False)
    md = chkpt['metadata']

    # Model
    model = model_class(config['model'], md)
    model.load_state_dict(chkpt['model_state_dict'])
    dtype = next(model.parameters()).dtype

    # Data transformations
    _data_transform_x = make_transform(md['config'].get('transform_x', None))
    _data_transform_u = make_transform(md['config'].get('transform_u', None))
    _data_transform_x.load_state_dict(md["transform_x_state"])
    _data_transform_u.load_state_dict(md["transform_u_state"])

    # Prediction in data space
    def predict_fn(x0, u, t, device="cpu"):
        """Predict trajectory in data space."""
        _x0 = _data_transform_x.transform([x0])[0]
        _x0 = torch.tensor(_x0, dtype=dtype, device=device)
        _u  = _data_transform_u.transform([u])[0]
        _u  = torch.tensor(_u, dtype=dtype, device=device)
        with torch.no_grad():
            pred = model.predict(_x0, _u, t).cpu().numpy()
        return _data_transform_x.inverse_transform([pred])[0]

    return model, predict_fn
