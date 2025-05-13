from pathlib import Path
import sys, logging
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    filename='train_node.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)  

from src.training.node_trainer import NODETrainer

if __name__ == "__main__":
    config_path = 'config_node.yaml'
    trainer = NODETrainer(config_path)
    trainer.train()