"""
Test cases for autonomous dynamics.

`ct`: Continuous time models, GLDM and GKBF, with NODE and weak form training.
`dt`: Discrete time models, DGLDM and DGKBF, with NODE training.

Also KBF/DKBF with linear training.
"""

import copy
import os
import pytest
import shutil
import torch

from dymad.io import load_model
from dymad.models import DKBF, DLDM, KBF, LDM
import dymad.training.driver as training_driver
from dymad.training import LinearTrainer, NODETrainer, StackedTrainer, WeakFormTrainer

mdl_kb = {
    "name" : 'kp_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "hidden_dimension" : 32,
    "koopman_dimension" : 4,
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "gain": 0.01}
mdl_ld = {
    "name": "kp_model",
    "encoder_layers": 0,
    "processor_layers": 2,
    "decoder_layers": 0,
    "hidden_dimension": 32,
    "activation": "prelu",
    "weight_init": "xavier_uniform",
    "gain": 0.01}
mdl_kl = {
    "name" : 'kp_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "hidden_dimension" : 32,
    "koopman_dimension" : 8,
    "activation" : "tanh",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform"}

crit_part = {
    "dynamics" : {"weight": 1.0},
    "recon" : {"weight": 1.0}
}
crit_full = {
    "dynamics" : {
        "type": "wmse",
        "weight": 1.0,
        "params": {
            "alpha": 0.5
        }},
    "recon" : {
        "type": "mse",
        "weight": 1.0,
        "params": {
            "reduction": "sum"
        }}
}
crit_pred = {
    "type": "wmse",
    "params": {
        "alpha": -0.5
    }}

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
    "ode_method": "dopri5",
    "ode_args": {
        "rtol": 1.e-7,
        "atol": 1.e-9}
    }
trn_dt = {
    "n_epochs": 10,
    "save_interval": 5,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [3, 5],
    "sweep_epoch_step": 5,
    "chop_mode": "initial"}
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "method": "truncated",
    "params": 2,
}
trn_phase = [copy.deepcopy(trn_wf),
            copy.deepcopy(trn_nd),]
trn_phase[0]["trainer"] = "Weak"
trn_phase[1]["trainer"] = "NODE"

cv = {
   "param_grid": {
        "model.hidden_dimension": [16, 32],
        "training.sweep_epoch_step": [3, 5]
    },
    "metric": "total"
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


def _mixed_schedule(trainer, base_cfg):
    chunk = copy.deepcopy(base_cfg)
    chunk["n_epochs"] = 3
    tail = copy.deepcopy(base_cfg)
    tail["n_epochs"] = 4
    return [
        _optimizer_phase(trainer, chunk),
        {
            "repeat": {
                "times": 2,
                "phases": [
                    _linear_solve_phase("truncated", params=2),
                    _optimizer_phase(trainer, chunk),
                ],
            }
        },
        _optimizer_phase(trainer, tail),
    ]

cfgs = [
    ('ldm_wf',    LDM,  WeakFormTrainer, {"model": mdl_ld, "criterion": crit_part, "training" : trn_wf}),
    ('ldm_node',  LDM,  NODETrainer,     {"model": mdl_ld, "criterion": crit_full, "training" : trn_nd}),
    ('kbf_wf',    KBF,  WeakFormTrainer, {"model": mdl_kb, "prediction_criterion": crit_pred, "training" : trn_wf}),
    ('kbf_node',  KBF,  StackedTrainer,  {"model": mdl_kb, "phases" : trn_phase}),
    ('kbf_wfls',  KBF,  StackedTrainer,  {"model": mdl_kb, "phases" : _mixed_schedule("Weak", trn_wf)}),
    ('kbf_ndls',  KBF,  StackedTrainer,  {"model": mdl_kb, "phases" : _mixed_schedule("NODE", trn_nd)}),
    ('kbf_ln',    KBF,  LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ('dldm_nd',   DLDM, NODETrainer,     {"model": mdl_ld, "training" : trn_dt}),
    ('dkbf_nd',   DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_dt, "cv": cv}),
    ('dkbf_ndls', DKBF, StackedTrainer,  {"model": mdl_kb, "phases" : _mixed_schedule("NODE", trn_dt)}),
    ('dkbf_ln',   DKBF, LinearTrainer,   {"model": mdl_kl, "training" : trn_ln}),
    ]

def train_case(idx, data, path, chkpt=None):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path/'kp_model.yaml'
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()

def predict_case(idx, sample, path):
    x_data, t_data = sample
    _, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path/'kp_model/kp_model.pt')
    with torch.no_grad():
        prd_func(x_data, t_data)

def test_non_linear_kp_workflow_routes_through_trainer_run(kp_data, env_setup, monkeypatch):
    calls = {"init": 0, "run": 0}
    real_trainer_run = training_driver.TrainerRun

    class _InstrumentedTrainerRun(real_trainer_run):
        def __init__(self, *args, **kwargs):
            calls["init"] += 1
            cfg = kwargs["config"]
            calls["phase_trainers"] = [phase["trainer"] for phase in cfg.get("phases", [])]
            calls["run_name"] = kwargs["run_name"]
            super().__init__(*args, **kwargs)

        def run(self, *args, **kwargs):
            calls["run"] += 1
            return super().run(*args, **kwargs)

    monkeypatch.setattr(training_driver, "TrainerRun", _InstrumentedTrainerRun)

    try:
        # idx=1 corresponds to NODE-based non-linear LDM workflow.
        train_case(1, kp_data, env_setup)
    finally:
        if os.path.exists(env_setup/'kp_model'):
            shutil.rmtree(env_setup/'kp_model')

    assert calls["init"] == 1
    assert calls["run"] == 1
    assert calls["phase_trainers"] == ["NODE"]
    assert calls["run_name"] == "kp_model_c0_f0"

@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_kp(kp_data, kp_test, env_setup, idx):
    train_case(idx, kp_data, env_setup)
    predict_case(idx, kp_test, env_setup)
    if os.path.exists(env_setup/'kp_model'):
        shutil.rmtree(env_setup/'kp_model')
