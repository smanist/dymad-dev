"""
Test cases for dynamics with inputs on graph.

`ct`: Continuous time models, GLDM and GKBF, with NODE and weak form training.
`dt`: Discrete time models, DGLDM and DGKBF, with NODE training.

Also GKBF/DGKBF with linear training.

Sweep mode included for NODE training.
"""

import copy
import os
import pytest
import torch

from dymad.models import DGKBF, DGLDM, GKBF, GLDM
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
from dymad.utils import load_model

mdl_kb = {
    "name" : 'ltg_model',
    "encoder_layers": 1,
    "decoder_layers": 1,
    "latent_dimension": 8,
    "koopman_dimension": 4,
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

ls_opt = {
    "method": "full",
    "interval": 3,
    "times": 2}
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
trn_wfls = copy.deepcopy(ls_opt)
trn_wfls.update(trn_wf)
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
trn_ndls = copy.deepcopy(ls_opt)
trn_ndls.update(trn_nd)
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
trn_dtls = copy.deepcopy(ls_opt)
trn_dtls.update(trn_dt)
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "ls_update": {
        "method": "full"
    }}

cfgs = [
    ('ldm_wf',    GLDM,  WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node',  GLDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',    GKBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node',  GKBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_wfls',  GKBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wfls}),
    ('kbf_ndls',  GKBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_ndls}),
    ('kbf_ln',    GKBF,  LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),
    ('dldm_nd',   DGLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',   DGKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ndls', DGKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dtls}),
    ('dkbf_ln',   DGKBF, LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),]

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

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_ltg(ltg_data, ltg_gau, env_setup, idx):
    train_case(idx, ltg_data, env_setup)
    predict_case(idx, ltg_gau, env_setup)
    if os.path.exists(env_setup/'ltg_model.pt'):
        os.remove(env_setup/'ltg_model.pt')
