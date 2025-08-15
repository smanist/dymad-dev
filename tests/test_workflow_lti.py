"""
Test cases for dynamics with inputs.

`ct`: Continuous time models, LDM and KBF, with NODE and weak form training.
`dl`: Continuous time models, LDM and KBF with delay, with NODE and weak form training.
`dt`: Discrete time models, DLDM and DKBF, with NODE training.  Chop mode included.

Also KBF/DKBF with linear training.

Sweep mode included for NODE training.
"""

import copy
import os
import pytest
import torch

from dymad.models import DKBF, DLDM, KBF, LDM
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer
from dymad.utils import load_model

trx = [
    {"type": "Scaler", "mode": "std"},
    {"type": "delay", "delay": 1}
]
tru = {
    "type": "delay",
    "delay": 1
}

mdl_kb = {
    "name" : 'lti_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 8,
    "koopman_dimension" : 4,
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "gain": 0.01}
mdl_ld = {
    "name": "lti_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "latent_dimension": 32,
    "activation": "prelu",
    "weight_init": "xavier_uniform",
    "gain": 0.01}
mdl_kl = {
    "name" : 'lti_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 4,
    "activation" : "none",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform"}

ls_opt = {
    "method": "truncated",
    "params": 2,
    "interval": 3,
    "times": 2,
    "start_with_ls": False}
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
    "sweep_lengths": [200, 501],
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
    "chop_mode": "unfold",
    "chop_step": 0.5,}
trn_dtls = copy.deepcopy(ls_opt)
trn_dtls.update(trn_dt)
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "ls_update": {
        "method": "truncated",
        "params": 2
    }}

cfgs = [
    ('ldm_nddl',  LDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd, "transform_x" : trx, "transform_u": tru}),
    ('kbf_wfdl',  KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf, "transform_x" : trx, "transform_u": tru}),
    ('ldm_wf',    LDM,  WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node',  LDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',    KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_wfls',  KBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wfls}),
    ('kbf_ndls',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_ndls}),
    ('kbf_ln',    KBF,  LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ('dldm_nd',   DLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',   DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ndls', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_ndls}),
    ('dkbf_ln',   DKBF, LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ]

IDX_DL = [0, 1]

def train_case(idx, data, path):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'lti_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path, ifdl = False):
    x_data, t_data, u_data = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'lti_model.pt', path/'lti_model.yaml', config_mod=opt)
    with torch.no_grad():
        if ifdl:
            prd_func(x_data, u_data, t_data[:-1])
        else:
            prd_func(x_data, u_data, t_data)

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_lti(lti_data, lti_gau, env_setup, idx):
    ifdl = idx in IDX_DL
    train_case(idx, lti_data, env_setup)
    predict_case(idx, lti_gau, env_setup, ifdl=ifdl)
    if os.path.exists(env_setup/'lti_model.pt'):
        os.remove(env_setup/'lti_model.pt')
