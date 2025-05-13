from pathlib import Path
import sys, yaml, logging, random, os
import torch
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    filename='train.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)  

from src.data.trajectory_manager import TrajectoryManager
from src.models.wMLP import weakFormMLP
from src.losses.weak_form import weak_form_loss
from src.losses.evaluation import prediction_rmse
from src.utils.plot import plot_hist
from src.utils.checkpoint import load_checkpoint, save_checkpoint

def train(dataloader, model, opt, scheduler, criterion, metadata, device, min_lr=5e-5):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        states = batch[:, :, :metadata['n_state_features']]
        controls = batch[:, :, -metadata['n_control_features']:]
        predictions = model(states, controls)
        loss = weak_form_loss(batch, predictions, metadata, criterion)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    scheduler.step()
    for param_group in opt.param_groups:
        if param_group['lr'] < min_lr:
            param_group['lr'] = min_lr

    return total_loss / len(dataloader)

def test(dataloader, model, criterion, metadata, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            states = batch[:, :, :metadata['n_state_features']]
            controls = batch[:, :, -metadata['n_control_features']:]
            predictions = model(states, controls)
            loss = weak_form_loss(batch, predictions, metadata, criterion)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Load the configuration file
    config_path = 'config_wmlp.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_name = config['model']['name']
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint_path = f'./checkpoints/{model_name}_checkpoint.pt'
    # Check if the checkpoint exists, if so, load the metadata and override the yaml config
    if os.path.exists(checkpoint_path) and config['training']['load_checkpoint']:
        logging.info(f"Checkpoint found at {checkpoint_path}, overriding the yaml config.")
        checkpoint = torch.load(checkpoint_path)
        metadata = checkpoint['metadata']
    else:
        logging.info(f"No checkpoint found at {checkpoint_path}, using the yaml config.")
        metadata = {'config': config}
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Double precision: {metadata['config']['data']['double_precision']}")
    # Initialize the TrajectoryManager
    tm = TrajectoryManager(metadata, device=device)
    dataloaders, datasets, metadata = tm.process_all()
    train_loader,validation_loader,test_loader = dataloaders
    train_set,validation_set,test_set = datasets
    # Initialize the model
    model = weakFormMLP(config['model'], metadata).to(device)
    if config['data']['double_precision']:
        model = model.double()
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.999)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Load checkpoint if it exists
    best_model_path = f'./{model_name}.pt'
    start_epoch, best_loss, hist, _ = load_checkpoint(model, opt, scheduler, checkpoint_path, config['training']['load_checkpoint'])
    # Train the model
    for epoch in range(start_epoch, config['training']['n_epochs']):
        train_loss = train(train_loader, model, opt, scheduler, criterion, metadata, device)
        val_loss = test(validation_loader, model, criterion, metadata, device)
        test_loader_loss = test(test_loader, model, criterion, metadata, device)
        hist.append([train_loss, val_loss, test_loader_loss])
        logging.info(f"Epoch {epoch+1}/{config['training']['n_epochs']}, "
                     f"Train Loss: {train_loss:.4f}, "
                     f"Validation Loss: {val_loss:.4f}, "
                     f"Test Loss: {test_loader_loss:.4f}")
        
        plot_hist(hist, epoch+1, model_name)
        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, opt, scheduler, epoch, best_loss, hist, metadata, best_model_path
            )
            logging.info(f"Best model saved at epoch {epoch+1} with validation loss {best_loss:.4f}")

        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(
                model, opt, scheduler, epoch, best_loss, hist, metadata, checkpoint_path
            )

            # Try prediction in a random trajectory from each set
            # Select a random trajectory from each dataset
            traj_train = random.choice(train_set)
            traj_val = random.choice(validation_set)
            traj_test = random.choice(test_set)
            ts = tm.t[0] ## TODO: this will not be correct when different length of trajectories exist in dataset
            
            # Call the prediction function on each trajectory (assuming each has 'truth' and 'ts' keys)
            rmse_train = prediction_rmse(model, traj_train, ts, metadata, model_name, plot=False)
            rmse_val = prediction_rmse(model, traj_val, ts, metadata, model_name, plot=False)
            rmse_test = prediction_rmse(model, traj_test, ts, metadata, model_name, plot=True)
        
            logging.info(
                f"Prediction RMSE - Train: {rmse_train:.4f}, Validation: {rmse_val:.4f}, Test: {rmse_test:.4f}"
            )

if __name__ == "__main__":
    main()