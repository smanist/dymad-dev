import torch, os, logging
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
        return 0, float("inf"), [], None

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not inference_mode:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        hist = checkpoint["hist"]
    
    return checkpoint["epoch"], checkpoint["best_loss"], hist, checkpoint["metadata"]

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, hist, metadata, checkpoint_path):
    """
    Save the model, optimizer, and scheduler states to a checkpoint file.
    
    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        scheduler: The scheduler to save.
        epoch (int): The current epoch number.
        best_loss (float): The best loss recorded so far.
        hist (list): The history of losses.
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
        "metadata": metadata,
    }, checkpoint_path)