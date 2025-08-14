"""
Test cases for dynamics with inputs on graph.

`ct`: Continuous time models, GLDM and GKBF, with NODE and weak form training.
`dt`: Discrete time models, DGLDM and DGKBF, with NODE training.

Also GKBF/DGKBF with linear training.

Sweep mode included for NODE training.
"""

import os
import torch

from dymad.models import DGKBF, DGLDM, GKBF, GLDM
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
from dymad.utils import load_model

mdl_kb = {
    "name" : 'ltg_model',
    "encoder_layers": 1,
    "decoder_layers": 1,
    "latent_dimension": 32,
    "koopman_dimension": 3,
    "const_term": True,
    "autoencoder_type": "cat",
    "gcl": "sage",
    "activation": "none",
    "weight_init": "xavier_uniform",
    "input_order": "cubic",
    "gain": 0.01}
mdl_ld = {
    "name": "ltg_model",
    "encoder_layers": 1,
    "processor_layers": 1,
    "decoder_layers": 1,
    "latent_dimension": 32,
    "autoencoder_type": "smp",
    "gcl": "sage",
    "activation": "none",
    "weight_init": "xavier_uniform",
    "input_order": "cubic",
    "gain": 0.01}

trn_wf = {
    "n_epochs": 10,
    "save_interval": 5,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "weak_form_params": {
        "N": 13,
        "dN": 2,
        "ordpol": 2,
        "ordint": 2}}
trn_nd = {
    "n_epochs": 10,
    "save_interval": 5,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [10, 20],
    "sweep_epoch_step": 5,
    "ode_method": "dopri5"}
trn_dt = {
    "n_epochs": 10,
    "save_interval": 5,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [3, 5],
    "sweep_epoch_step": 5,
    "chop_mode": "initial"}
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "method": "full"}

cfgs = [
    ('ldm_wf',   GLDM,  WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node', GLDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',   GKBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', GKBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_ln',   GKBF,  LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),
    ('dldm_nd',  DGLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',  DGKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ln',  DGKBF, LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),]

IDX_CT = [0, 1, 2, 3]
IDX_DT = [5, 6, 7]

def train_case(idx, data, path):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'ltg_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path):
    x_data, t_data, u_data, edge_index = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'ltg_model.pt', path/'ltg_model.yaml', config_mod=opt)
    with torch.no_grad():
        prd_func(x_data, u_data, t_data, ei=edge_index)

def test_ltg_ct(ltg_data, ltg_gau, env_setup):
    for _i in IDX_CT:
        train_case(_i, ltg_data, env_setup)
        predict_case(_i, ltg_gau, env_setup)
    os.remove(env_setup/'ltg_model.pt')

def test_ltg_ln(ltg_data, ltg_gau, env_setup):
    for _i in [4]:
        train_case(_i, ltg_data, env_setup)
        predict_case(_i, ltg_gau, env_setup)
    os.remove(env_setup/'ltg_model.pt')

def test_ltg_dt(ltg_data, ltg_gau, env_setup):
    for _i in IDX_DT:
        train_case(_i, ltg_data, env_setup)
        predict_case(_i, ltg_gau, env_setup)
    os.remove(env_setup/'ltg_model.pt')
