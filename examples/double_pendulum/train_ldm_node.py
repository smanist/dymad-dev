from pathlib import Path
import sys, logging
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    filename='train_ldm_node.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)  

from src.training.node_trainer import NODETrainer

if __name__ == "__main__":
    config_path = 'config_ldm_node.yaml'
    
    logging.info(f"Training LDM with NODE approach")
    logging.info(f"Config: {config_path}")
    
    # Create trainer - TrainerBase handles all the setup
    trainer = NODETrainer(config_path)
    
    logging.info(f"Model: {trainer.model_name}")
    logging.info(f"Latent dimension: {trainer.config['model']['latent_dimension']}")
    logging.info(f"Architecture: {trainer.config['model']['encoder_layers']}-{trainer.config['model']['processor_layers']}-{trainer.config['model']['decoder_layers']}")
    logging.info(f"Training epochs: {trainer.config['training']['n_epochs']}")
    logging.info(f"ODE method: {trainer.ode_method}")
    logging.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    logging.info(f"Device: {trainer.device}")
    
    # Train the model
    trainer.train()