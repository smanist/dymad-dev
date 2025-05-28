from pathlib import Path
import sys, logging
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    filename='train_wgkbf.log',  
    filemode='w',  
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)  

import torch
from torch_geometric import utils as pyg_utils
from src.training.wgkbf_trainer import wGKBFTrainer

if __name__ == "__main__":

    edgeIdx = torch.LongTensor([
        [5,0,1,2,24,3,2,4,5,6,11,11,11,7,8,9,11,12,14,14,13,18,15,16,15,19,20,20,21,22,21,23,21,10,17],
        [0,2,0,5,3,25,4,3,26,11,7,8,9,9,9,10,12,14,27,13,19,15,16,20,17,18,21,22,22,1,23,28,28,29,30]
    ])

    # Node indices based on types (converted to 0-indexing)
    thermal_states = [i-1 for i in [1,2,3,4,5,6,10,11,18,23]]
    electrical_states = [i-1 for i in [7,8,9,12,13,14,15,16,17,19,20,21]]
    mechanical_states = [i-1 for i in [22, 24]]
    _nStates = sum([len(_states) for _states in [thermal_states,electrical_states,mechanical_states]])
    # edgeIdx
    _mask = edgeIdx<_nStates
    _mask = _mask[0] & _mask[1]
    edgeIdx = edgeIdx[:,_mask]
    edgeIdx = pyg_utils.to_undirected(edgeIdx) # undirected graph
    edgeIdx = pyg_utils.add_self_loops(edgeIdx)[0] # self-connection

    adj = pyg_utils.to_dense_adj(edgeIdx)[0]

    config_path = 'config_wgkbf.yaml'
    trainer = wGKBFTrainer(config_path, adj)
    trainer.train()