from pathlib import Path
import sys, logging
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    filename='train_kbf.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)  

from src.training.kbf_trainer import KBFTrainer

if __name__ == "__main__":
    config_path = 'config_kbf_weak_form.yaml'
    trainer = KBFTrainer(config_path)
    trainer.train() 