"""
Test cases for autonomous dynamics.

`ct`: Continuous time models, GLDM and GKBF, with NODE and weak form training.
`dt`: Discrete time models, DGLDM and DGKBF, with NODE training.
`rst`: Restart training from checkpoint.

Also KBF/DKBF with linear training.
"""

import copy
import os
import pytest
import torch

from dymad.models import DKBF, DLDM, KBF, LDM
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
from dymad.utils import load_model

mdl_kb = {
    "name" : 'kp_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 32,
    "koopman_dimension" : 4,
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "gain": 0.01}
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "latent_dimension": 32,
    "activation": "prelu",
    "weight_init": "xavier_uniform",
    "gain": 0.01}
mdl_kl = {
    "name" : 'kp_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 8,
    "activation" : "tanh",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform"}

ls_opt = {
    "method": "truncated",
    "params": 2,
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
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9}
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
        "method": "truncated",
        "params": 2
    }}

cfgs = [
    ('ldm_wf',    LDM,  WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node',  LDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',    KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_wfls',  KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wfls}),
    ('kbf_ndls',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_ndls}),
    ('kbf_ln',    KBF,  LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ('dldm_nd',   DLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',   DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ndls', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dtls}),
    ('dkbf_ln',   DKBF, LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ]

def train_case(idx, data, path, chkpt=None):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    if chkpt is not None:
        opt["training"]["load_checkpoint"] = path/chkpt
    config_path = path/'kp_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path):
    x_data, t_data = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'kp_model.pt', path/'kp_model.yaml', config_mod=opt)
    with torch.no_grad():
        prd_func(x_data, t_data)

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_kp(kp_data, kp_test, env_setup, idx):
    train_case(idx, kp_data, env_setup)
    predict_case(idx, kp_test, env_setup)
    if os.path.exists(env_setup/'kp_model.pt'):
        os.remove(env_setup/'kp_model.pt')

def test_kp_rst(kp_data, kp_test, env_setup):
    train_case(0, kp_data, env_setup)
    train_case(1, kp_data, env_setup, chkpt='checkpoints/kp_model_checkpoint.pt')
    predict_case(1, kp_test, env_setup)
    if os.path.exists(env_setup/'kp_model.pt'):
        os.remove(env_setup/'kp_model.pt')
