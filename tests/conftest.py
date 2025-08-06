import numpy as np
import os
from pathlib import Path
import pytest
import shutil

from dymad.utils import TrajectorySampler

HERE = Path(__file__).parent

A = np.array([
            [0., 1.],
            [-1., -0.1]])
def f(t, x, u):
    return (x @ A.T) + u
g = lambda t, x, u: x

config_chr = {
    "control" : {
        "kind": "chirp",
        "params": {
            "t1": 4.0,
            "freq_range": (0.5, 2.0),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0)}}}

config_gau = {
    "control" : {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std":  1.0,
            "t1":   4.0,
            "dt":   0.2,
            "mode": "zoh"}}}

@pytest.fixture(scope='session')
def env_setup():
    # ---- runs ONCE before any tests execute ----

    # ---- Interface to the tests ----
    yield HERE

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    shutil.rmtree(HERE/'results', ignore_errors=True)
    shutil.rmtree(HERE/'checkpoints', ignore_errors=True)

@pytest.fixture(scope='session')
def lti_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 128
    N = 501
    t_grid = np.linspace(0, 5, N)

    sampler = TrajectorySampler(f, g, config=HERE/'lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(HERE/'lti.npz', t=ts, x=ys, u=us)

    # ---- Interface to the tests ----
    yield HERE/'lti.npz'

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE/'lti.npz'):
        os.remove(HERE/'lti.npz')

@pytest.fixture(scope='session')
def lti_gau():
    # ---- runs ONCE before any tests execute ----
    N = 501
    t_grid = np.linspace(0, 5, N)
    sampler = TrajectorySampler(f, g, config=HERE/'lti_data.yaml', config_mod=config_gau)
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    # ---- Interface to the tests ----
    yield (x_data, t_data, u_data)

@pytest.fixture(scope='session')
def ltg_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 128
    N = 501
    t_grid = np.linspace(0, 5, N)

    sampler = TrajectorySampler(f, g, config=HERE/'lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)

    # Pretending a 3-node graph
    np.savez_compressed(HERE/'ltg.npz', t=ts, x=np.concatenate([ys, ys, ys], axis=-1), u=np.concatenate([us, us, us], axis=-1))

    # ---- Interface to the tests ----
    yield HERE/'ltg.npz'

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE/'ltg.npz'):
        os.remove(HERE/'ltg.npz')
