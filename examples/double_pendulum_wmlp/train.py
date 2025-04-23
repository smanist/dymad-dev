from pathlib import Path
import sys, yaml, logging, random
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
from src.losses.weak_form import weak_form_loss, prediction_rmse
from src.utils.plot import plot_hist
from src.utils.checkpoint import load_checkpoint, save_checkpoint

def train(dataloader, model, opt, scheduler, criterion, data_meta, device, min_lr=5e-5):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        states = batch[:, :, :data_meta['n_state_features']]
        controls = batch[:, :, -data_meta['n_control_features']:]
        predictions = model(states, controls)
        loss = weak_form_loss(batch, predictions, data_meta, criterion)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    scheduler.step()
    for param_group in opt.param_groups:
        if param_group['lr'] < min_lr:
            param_group['lr'] = min_lr

    return total_loss / len(dataloader)


def test(dataloader, model, criterion, data_meta, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            states = batch[:, :, :data_meta['n_state_features']]
            controls = batch[:, :, -data_meta['n_control_features']:]
            predictions = model(states, controls)
            loss = weak_form_loss(batch, predictions, data_meta, criterion)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():

    config_path = 'config.yaml'

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the TrajectoryManager
    tm = TrajectoryManager(config, device=device)

    dataloaders, datasets, data_meta = tm.process_all()
    train_loader,validation_loader,test_loader = dataloaders
    train_set,validation_set,test_set = datasets

    # Initialize model
    model_name = config['model']['name']
    model = weakFormMLP(config['model'], data_meta).double().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.999)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Load checkpoint if it exists
    checkpoint_path = f'./checkpoints/{model_name}_checkpoint.pt'
    best_model_path = f'./{model_name}.pt'
    start_epoch, best_loss, hist = load_checkpoint(model, opt, scheduler, checkpoint_path, config['training']['load_checkpoint'])

    for epoch in range(start_epoch, config['training']['n_epochs']):
        train_loss = train(train_loader, model, opt, scheduler, criterion, data_meta, device)
        val_loss = test(validation_loader, model, criterion, data_meta, device)
        test_loader_loss = test(test_loader, model, criterion, data_meta, device)
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
                model, opt, scheduler, epoch, best_loss, hist, best_model_path
            )
            logging.info(f"Best model saved at epoch {epoch+1} with validation loss {best_loss:.4f}")

        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(
                model, opt, scheduler, epoch, best_loss, hist, checkpoint_path
            )

            # Try prediction in a random trajectory from each set
            # Select a random trajectory from each dataset
            traj_train = random.choice(train_set)
            traj_val = random.choice(validation_set)
            traj_test = random.choice(test_set)
            ts = tm.t[0] ## TODO: this will not be correct when different length of trajectories exist in dataset
            
            # Call the prediction function on each trajectory (assuming each has 'truth' and 'ts' keys)
            rmse_train = prediction_rmse(model, traj_train, ts, data_meta, plot=False)
            rmse_val = prediction_rmse(model, traj_val, ts, data_meta, plot=False)
            rmse_test = prediction_rmse(model, traj_test, ts, data_meta, plot=True)
        
            logging.info(
                f"Prediction RMSE - Train: {rmse_train:.4f}, Validation: {rmse_val:.4f}, Test: {rmse_test:.4f}"
            )

if __name__ == "__main__":
    main()