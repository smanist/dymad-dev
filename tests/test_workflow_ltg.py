"""
Test cases for dynamics with inputs on graph.

`ct`: Continuous time models, GLDM and GKBF, with NODE and weak form training.
`dt`: Discrete time models, DGLDM and DGKBF, with NODE training.

Also GKBF/GKM/DGKBF/DGKM/DGKMSK with linear training.

Sweep mode included for NODE training.
"""

import copy
import os
import pytest
import shutil
import torch

from dymad.io import load_model
from dymad.models import DGKBF, DGKM, DGKMSK, DGLDM, DGLTI, DSDMG, GKBF, GKM, GLDM, GLTI
from dymad.training import WeakFormTrainer, NODETrainer, LinearTrainer

trx = [
    {"type": "scaler", "mode": "std"},
    {"type": "delay", "delay": 1}
]
tru = {
    "type": "delay",
    "delay": 1
}

mdl_kb = {
    "name" : 'ltg_model',
    "encoder_layers": 1,
    "decoder_layers": 1,
    "hidden_dimension": 8,
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
    "hidden_dimension": 32,
    "autoencoder_type": "smp",
    "gcl": "sage",
    "activation": "none",
    "weight_init": "xavier_uniform",
    "input_order": "cubic",
    "gain": 0.01}
mdl_km = {
    "name": "ltg_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "kernel_dimension": 2,
    "input_order": "cubic",
    "type": "share",
    "kernel": {
        "type": "sc_rbf",
        "input_dim": 2,
        "lengthscale_init": 1.0
    },
    "ridge_init": 1.e-4}
mdl_sd = {
    "name" : 'ltg_model',
    "autoencoder_type" : "seq_smp",
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "processor_type" : "gnn_smp",
    "processor_layers": 1,
    "hidden_dimension" : 8,
    "latent_dimension" : 8,
    "activation" : "none",
    "weight_init" : "xavier_uniform"}

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
    "ls_update": {
        "method": "full"
    }}

cfgs = [
    ('ldm_nddl',  GLDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd, "transform_x" : trx, "transform_u": tru}),
    ('sdm_smp',   DSDMG, NODETrainer,     {"model": mdl_sd, "training" : trn_dt, "transform_x" : trx, "transform_u": tru}),
    ('ldm_wf',    GLDM,  WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node',  GLDM,  NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',    GKBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node',  GLTI,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_wfls',  GKBF,  WeakFormTrainer, {"model": mdl_kb, "training" : trn_wfls}),
    ('kbf_ndls',  GKBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_ndls}),
    ('kbf_ln',    GKBF,  LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),
    ('km_ln',     GKM,   LinearTrainer,   {"model": mdl_km, "training" : trn_ln}),
    ('dldm_nd',   DGLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',   DGKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt}),
    ('dkbf_ndls', DGKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dtls}),
    ('dkbf_ln',   DGLTI, LinearTrainer,   {"model": mdl_kb, "training" : trn_ln}),
    ('dkm_ln',    DGKM,  LinearTrainer,   {"model": mdl_km, "training" : trn_ln}),
    ('dkmsk_ln',  DGKMSK,LinearTrainer,   {"model": mdl_km, "training" : trn_ln}),
    ]

IDX_DL = [0, 1]

def train_case(idx, data, path):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'ltg_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path, ifdl = False):
    x_data, t_data, u_data, edge_index = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'ltg_model/ltg_model.pt')
    with torch.no_grad():
        if ifdl:
            prd_func(x_data, t_data[:-1], u=u_data, ei=torch.tensor(edge_index))
        else:
            prd_func(x_data, t_data, u=u_data, ei=torch.tensor(edge_index))

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_ltg(ltg_data, ltg_gau, env_setup, idx):
    ifdl = idx in IDX_DL
    train_case(idx, ltg_data, env_setup)
    predict_case(idx, ltg_gau, env_setup, ifdl=ifdl)
    if os.path.exists(env_setup/'ltg_model'):
        shutil.rmtree(env_setup/'ltg_model')
