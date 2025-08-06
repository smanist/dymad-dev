import os
import torch

from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model


mdl_kb = {
    "name" : 'lti_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 32,
    "koopman_dimension" : 4,
    "activation" : "prelu",
    "weight_init" : "xavier_uniform"}
mdl_ld = {
    "name": "lti_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "latent_dimension": 32,
    "activation": "prelu",
    "weight_init": "xavier_uniform"}

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
    "sweep_lengths": [30, 50, 100, 200, 301],
    "sweep_epoch_step": 400,
    "ode_method": "dopri5",
    "rtol": 1e-7,
    "atol": 1e-9
}

cfgs = [
    ('ldm_wf',   LDM, WeakFormTrainer, {"model": mdl_ld, "training" : trn_wf}),
    ('ldm_node', LDM, NODETrainer,     {"model": mdl_ld, "training" : trn_nd}),
    ('kbf_wf',   KBF, WeakFormTrainer, {"model": mdl_kb, "training" : trn_wf}),
    ('kbf_node', KBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ]

IDX = [0, 1, 2, 3]

def train_case(idx, data, path):
    mdl, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'lti_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path):
    x_data, t_data, u_data = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'lti_model.pt', path/'lti_model.yaml', config_mod=opt)
    with torch.no_grad():
        prd_func(x_data, u_data, t_data)

def test_cases(lti_data, lti_gau, env_setup):
    for _i in IDX:
        train_case(_i, lti_data, env_setup)
        predict_case(_i, lti_gau, env_setup)
    os.remove(env_setup/'lti_model.pt')
