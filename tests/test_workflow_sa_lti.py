import copy
import os
import shutil

import numpy as np
import pytest
import torch

from dymad.exec.workflow import CompatibilityExecutor
from dymad.io import load_model
from dymad.models import DKBF, KBF
from dymad.sako import SpectralAnalysis, SpectralAnalysisAdapter, SpectralPlottingAdapter
from dymad.training import LinearTrainer, StackedTrainer

mdl_kb = {
    "name": "sa_model",
    "encoder_layers": 1,
    "decoder_layers": 1,
    "koopman_dimension": 16,
    "activation": "none",
    "autoencoder_type": "cat",
    "weight_init": "xavier_uniform",
    "predictor_type": "exp",
}

mdl_kl = {
    "name": "sa_model",
    "encoder_layers": 0,
    "decoder_layers": 0,
    "koopman_dimension": 16,
    "activation": "none",
    "weight_init": "xavier_uniform",
    "predictor_type": "exp",
}
trn_kl = [{"type": "scaler", "mode": "-11"}, {"type": "lift", "fobs": "poly", "Ks": [4, 4]}]

trn_nd1 = {
    "n_epochs": 100,
    "save_interval": 50,
    "load_checkpoint": False,
    "learning_rate": 5e-3,
    "decay_rate": 0.999,
    "sweep_lengths": [2, 4],
    "sweep_epoch_step": 100,
}
trn_nd2 = copy.deepcopy(trn_nd1)

trn_dt1 = copy.deepcopy(trn_nd1)
trn_dt2 = copy.deepcopy(trn_nd1)
trn_dt3 = copy.deepcopy(trn_nd1)

ref = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
}
trn_ln = {
    "method": "full",
}
trn_ln.update(ref)
trn_tr = {
    "method": "truncated",
    "params": 0.999,
}
trn_tr.update(ref)
trn_sa = {
    "method": "sako",
    "params": 9,
    "kwargs": {"remove_one": True},
}
trn_sa.update(ref)


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


def _spectral_schedule(method, params=None, *, kwargs=None):
    chunk = copy.deepcopy(trn_nd1)
    chunk["n_epochs"] = 50
    return [
        {
            "repeat": {
                "times": 2,
                "phases": [
                    _linear_solve_phase(method, params=params, kwargs=kwargs),
                    _optimizer_phase("NODE", chunk),
                ],
            }
        },
        _linear_solve_phase(method, params=params, kwargs=kwargs),
    ]


config_path = "sa_model.yaml"

dt = 0.5

cfgs = [
    (
        "kbf_nd1",
        KBF,
        StackedTrainer,
        {"model": mdl_kb, "phases": _spectral_schedule("truncated_log", params=2)},
    ),
    ("kbf_nd2", KBF, StackedTrainer, {"model": mdl_kb, "phases": _spectral_schedule("full_log")}),
    ("dkbf_nd1", DKBF, StackedTrainer, {"model": mdl_kb, "phases": _spectral_schedule("full")}),
    (
        "dkbf_nd2",
        DKBF,
        StackedTrainer,
        {"model": mdl_kb, "phases": _spectral_schedule("truncated", params=2)},
    ),
    (
        "dkbf_nd3",
        DKBF,
        StackedTrainer,
        {"model": mdl_kb, "phases": _spectral_schedule("sako", params=2)},
    ),
    ("dkbf_ln", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_ln}),
    ("dkbf_tr", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_tr}),
    ("dkbf_sa", DKBF, LinearTrainer, {"model": mdl_kl, "transform_x": trn_kl, "training": trn_sa}),
]


def train_case(idx, data, path):
    _, MDL, Trainer, opt = cfgs[idx]
    opt.update({"data": {"path": data}})
    config_path = path / "sa_model.yaml"
    trainer = Trainer(config_path, MDL, config_mod=opt)
    trainer.train()


def predict_case(idx, sample, path):
    x_data, t_data = sample
    mdl, MDL, _, opt = cfgs[idx]
    _, prd_func = load_model(MDL, path / "sa_model/sa_model.pt")
    with torch.no_grad():
        _prd = prd_func(x_data, t_data)
        _err = np.linalg.norm(_prd - x_data) / np.linalg.norm(x_data)

        if mdl in ["kbf_nd1", "dkbf_nd2", "dkbf_nd3"]:
            assert _err < 1e-4
        elif mdl in ["kbf_nd2", "dkbf_nd1"]:
            assert _err < 0.01
        elif mdl == "dkbf_tr":
            assert _err < 0.08
        else:
            assert _err < 2e-5


def sa_case(idx, path):
    _, MDL, _, _ = cfgs[idx]
    _s = SpectralAnalysis(MDL, path / "sa_model/sa_model.pt", dt=dt, reps=1e-10, etol=1e-12)

    xs = np.linspace(-1.3, 1.3, 4)
    gg = np.vstack([xs, xs])

    grid, _pss = _s.estimate_ps(gg, mode="disc", method="standard", return_vec=False)
    grid, _psk = _s.estimate_ps(gg, mode="disc", method="sako", return_vec=False)
    grid, _pss = _s.estimate_ps(gg, mode="cont", method="standard", return_vec=False)
    grid, _psk = _s.estimate_ps(gg, mode="cont", method="sako", return_vec=False)

    def func_obs(x):
        _x1, _x2 = x.T
        return _x1 + _x2

    _s.estimate_measure(func_obs, 6, 0.1, thetas=5)

    _s.eval_eigfunc_jac()
    _s.eval_eigmode_jac()


@pytest.mark.parametrize("idx", range(len(cfgs)))
def test_sa(sa_lti_data, sa_lti_test, env_setup, idx):
    train_case(idx, sa_lti_data, env_setup)
    predict_case(idx, sa_lti_test, env_setup)
    sa_case(idx, env_setup)
    if os.path.exists(env_setup / "sa_model"):
        shutil.rmtree(env_setup / "sa_model")


def test_spectral_analysis_routes_pseudospectrum_through_adapter(
    sa_lti_data, env_setup, monkeypatch
):
    train_case(5, sa_lti_data, env_setup)

    call_counter = {"estimate_ps": 0}
    original_estimate_ps = SpectralAnalysisAdapter.estimate_ps

    def wrapped_estimate_ps(self, *args, **kwargs):
        call_counter["estimate_ps"] += 1
        return original_estimate_ps(self, *args, **kwargs)

    monkeypatch.setattr(SpectralAnalysisAdapter, "estimate_ps", wrapped_estimate_ps)

    _, model_class, _, _ = cfgs[5]
    analysis = SpectralAnalysis(
        model_class,
        env_setup / "sa_model/sa_model.pt",
        dt=dt,
        reps=1e-10,
        etol=1e-12,
    )

    xs = np.linspace(-1.3, 1.3, 4)
    grid = np.vstack([xs, xs])
    analysis.estimate_ps(grid, mode="disc", method="standard", return_vec=False)

    assert call_counter["estimate_ps"] >= 1

    if os.path.exists(env_setup / "sa_model"):
        shutil.rmtree(env_setup / "sa_model")


def test_spectral_analysis_routes_plotting_through_adapter(sa_lti_data, env_setup, monkeypatch):
    train_case(5, sa_lti_data, env_setup)

    call_counter = {"plot_eigs": 0}

    def wrapped_plot_eigs(self, *args, **kwargs):
        call_counter["plot_eigs"] += 1
        return ("fig", "ax", [])

    monkeypatch.setattr(SpectralPlottingAdapter, "plot_eigs", wrapped_plot_eigs)

    _, model_class, _, _ = cfgs[5]
    analysis = SpectralAnalysis(
        model_class,
        env_setup / "sa_model/sa_model.pt",
        dt=dt,
        reps=1e-10,
        etol=1e-12,
    )

    result = analysis.plot_eigs()

    assert call_counter["plot_eigs"] >= 1
    assert result == ("fig", "ax", [])

    if os.path.exists(env_setup / "sa_model"):
        shutil.rmtree(env_setup / "sa_model")


def test_spectral_analysis_routes_snapshot_handle_flow_through_exec(
    sa_lti_data, env_setup, monkeypatch
):
    train_case(5, sa_lti_data, env_setup)

    call_counter = {"plan_spectral": 0, "materialize_spectral": 0}
    original_plan_spectral = CompatibilityExecutor.plan_spectral_analysis
    original_materialize_spectral = CompatibilityExecutor.materialize_spectral_adapter

    def wrapped_plan_spectral(self, *args, **kwargs):
        call_counter["plan_spectral"] += 1
        return original_plan_spectral(self, *args, **kwargs)

    def wrapped_materialize_spectral(self, *args, **kwargs):
        call_counter["materialize_spectral"] += 1
        return original_materialize_spectral(self, *args, **kwargs)

    monkeypatch.setattr(CompatibilityExecutor, "plan_spectral_analysis", wrapped_plan_spectral)
    monkeypatch.setattr(
        CompatibilityExecutor, "materialize_spectral_adapter", wrapped_materialize_spectral
    )

    _, model_class, _, _ = cfgs[5]
    analysis = SpectralAnalysis(
        model_class,
        env_setup / "sa_model/sa_model.pt",
        dt=dt,
        reps=1e-10,
        etol=1e-12,
    )
    analysis.estimate_ps(
        np.vstack([np.linspace(-1.0, 1.0, 3)] * 2), mode="disc", method="standard", return_vec=False
    )

    assert call_counter["plan_spectral"] >= 1
    assert call_counter["materialize_spectral"] >= 1

    if os.path.exists(env_setup / "sa_model"):
        shutil.rmtree(env_setup / "sa_model")
