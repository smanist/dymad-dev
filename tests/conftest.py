import numpy as np
from pathlib import Path
import pytest

from dymad.utils import TrajectorySampler

HERE = Path(__file__).parent

@pytest.fixture(scope='session')
def lti_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 128
    N = 501
    t_grid = np.linspace(0, 5, N)

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

    sampler = TrajectorySampler(f, g, config=HERE/'lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(HERE/'lti.npz', t=ts, x=ys, u=us)

    # ---- Interface to the tests ----
    yield HERE/'lti.npz'

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    import os
    if os.path.exists(HERE/'lti.npz'):
        os.remove(HERE/'lti.npz')

@pytest.fixture(scope='session')
def ltg_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 128
    N = 501
    t_grid = np.linspace(0, 5, N)

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

    sampler = TrajectorySampler(f, g, config=HERE/'lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)

    # Pretending a 3-node graph
    np.savez_compressed(HERE/'lti.npz', t=ts, x=np.concatenate([ys, ys, ys], axis=-1), u=np.concatenate([us, us, us], axis=-1))

    # ---- Interface to the tests ----
    yield HERE/'lti.npz'

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    import os
    if os.path.exists(HERE/'lti.npz'):
        os.remove(HERE/'lti.npz')
