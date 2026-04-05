"""
Test cases for kernel dynamics without inputs.
"""

import copy
import numpy as np
import os
import pytest
import shutil
import torch

from dymad.io import load_model
from dymad.models import DKM, DKMSK, KM, KMM
from dymad.training import LinearTrainer, NODETrainer, StackedTrainer

# Options
## Regular kernels
RIDGE = 1e-10
opt_rbf1 = {
    "type": "sc_rbf",
    "input_dim": 2,
    "lengthscale_init": 1.0
}
opt_rbf2 = {
    "type": "sc_rbf",
    "input_dim": 2,
    "lengthscale_init": None
}
opt_opk1 = {
    "type": "op_sep",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": [opt_rbf1],
    "Ls": np.array([[[1, 0], [0, 1]]])
}
opt_share = {
    "type": "share",
    "kernel": opt_rbf1,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
opt_indep = {
    "type": "indep",
    "kernel": [opt_rbf1, opt_rbf1],
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
opt_opval = {
    "type": "opval",
    "kernel": opt_opk1,
    "dtype": torch.float64,
    "ridge_init": RIDGE,
}

mdl_ref = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "kernel_dimension" : 2,
    }
mdl_share = copy.deepcopy(mdl_ref)
mdl_share.update(**opt_share)

mdl_indep = copy.deepcopy(mdl_ref)
mdl_indep.update(**opt_indep)

mdl_opval = copy.deepcopy(mdl_ref)
mdl_opval.update(**opt_opval)

## Manifold case
opt_opk = {
    "type": "op_tan",
    "input_dim": 2,
    "output_dim": 2,
    "kopts": opt_rbf2
}
opt_tange = {
    "type": "tangent",
    "kernel": opt_opk,
    "dtype": torch.float64,
    "ridge_init": RIDGE
}
mdl_mn = {
    "name" : 'ker_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "kernel_dimension" : 2,
    "manifold" : {
        "d" : 2,
        "T" : 3,
        "g" : 3
    }}
mdl_mn.update(**opt_tange)

# Training options
trn_ln = {
    "n_epochs": 1,
    "save_interval": 100,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "method": "raw",
}
trn_l1 = copy.deepcopy(trn_ln)
trn_l1["kwargs"] = {"order": 1}
trn_dt = {
    "n_epochs": 5,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
}


def _optimizer_phase(trainer, cfg):
    phase = copy.deepcopy(cfg)
    phase.update({"type": "optimizer", "trainer": trainer})
    return phase


def _linear_solve_phase(method, params=None, *, kwargs=None, reset_optimizer=True):
    phase = {"type": "linear_solve", "method": method, "reset_optimizer": reset_optimizer}
    if params is not None:
        phase["params"] = params
    if kwargs:
        phase["kwargs"] = copy.deepcopy(kwargs)
    return phase


trn_dt_phases = [
    {
        "repeat": {
            "times": 1,
            "phases": [
                _linear_solve_phase("raw", reset_optimizer=False),
                _optimizer_phase("NODE", trn_dt),
            ],
        }
    },
    _linear_solve_phase("raw", reset_optimizer=False),
]

cfgs = [
    ('km_ln',  KM,     LinearTrainer,     {"model": mdl_indep, "training" : trn_ln}),
    ('dkm_ln', DKM,    LinearTrainer,     {"model": mdl_opval, "training" : trn_ln}),
    ('dks_ln', DKMSK,  LinearTrainer,     {"model": mdl_share, "training" : trn_ln}),
    ('dks_nd', DKMSK,  StackedTrainer,    {"model": mdl_share, "phases" : trn_dt_phases}),
    ('kmm_ln', KMM,    LinearTrainer,     {"model": mdl_mn,    "training" : trn_l1}),
    ]

def train_case(idx, data, path):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'ker_model_auto.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path):
    x_data, t_data = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'ker_model/ker_model.pt')
    with torch.no_grad():
        prd_func(x_data, t_data)

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_ker(kp_data, kp_test, env_setup, idx):
    train_case(idx, kp_data, env_setup)
    predict_case(idx, kp_test, env_setup)
    if os.path.exists(env_setup/'ker_model'):
        shutil.rmtree(env_setup/'ker_model')
