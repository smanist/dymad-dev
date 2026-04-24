import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dymad.utils import TrajectorySampler, adj_to_edge

HERE = Path(__file__).parent
ALLOWED_TEST_FILE_PREFIXES = (
    "test_assert_",
    "test_workflow_",
    "test_slow_",
    "test_contract_",
    "test_agent_",
)
LTI_DATA_SEED = 12345
LTI_TEST_SEED = 12346
KP_DATA_SEED = 12347
KP_TEST_SEED = 12348
LTG_DATA_SEED = 12349
LTG_TEST_SEED = 12350
LTGA_DATA_SEED = 12351
LTGA_TEST_SEED = 12352
SA_LTI_DATA_SEED = 123
SA_LTI_TEST_SEED = 456


def pytest_addoption(parser):
    parser.addoption(
        "--record-baselines",
        action="store_true",
        default=False,
        help="Record baseline metrics for slow regression tests instead of comparing against them.",
    )


def pytest_collection_modifyitems(config, items):
    del config
    invalid = sorted(
        {
            Path(item.fspath).name
            for item in items
            if not Path(item.fspath).name.startswith(ALLOWED_TEST_FILE_PREFIXES)
        }
    )
    if invalid:
        allowed = ", ".join(ALLOWED_TEST_FILE_PREFIXES)
        invalid_list = ", ".join(invalid)
        raise pytest.UsageError(
            "Test file names must use one of the approved prefixes "
            f"({allowed}). Offending files: {invalid_list}"
        )


A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


def f_auto(t, x):
    return x @ A.T


def g_auto(t, x):
    return x


mu = -0.5
lm = -3


def f_kp(t, x):
    _d = np.array([mu * x[0], lm * (x[1] - x[0] ** 2)])
    return _d


adj = np.array([[0, 1, 2], [1, 0, 1], [1, 2, 0]])
edge_index = adj_to_edge(adj)[0]

config_chr = {
    "control": {
        "kind": "chirp",
        "params": {
            "t1": 4.0,
            "freq_range": (0.5, 2.0),
            "amp_range": (0.5, 1.0),
            "phase_range": (0.0, 360.0),
        },
    }
}

config_gau = {
    "control": {
        "kind": "gaussian",
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}


@pytest.fixture(scope="session")
def env_setup():
    # ---- runs ONCE before any tests execute ----

    # ---- Interface to the tests ----
    yield HERE

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    shutil.rmtree(HERE / "results", ignore_errors=True)
    shutil.rmtree(HERE / "checkpoints", ignore_errors=True)


@pytest.fixture(scope="session")
def trj_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 8
    N = 21
    t_grid = np.linspace(0, 0.1, N)

    sampler = TrajectorySampler(
        f,
        g,
        config=HERE / "lti_data.yaml",
        rng=LTI_DATA_SEED,
        config_mod=config_chr,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    ys = np.hstack([xs[0], xs[1]])  # This will be repeated along batch
    us = us[:, 0, :].reshape(B, 1, -1)  # This will be repeated along time
    ps = [np.arange(4) + _i for _i in range(len(xs))]
    np.savez_compressed(HERE / "trj.npz", t=ts, x=xs, y=ys, u=us, p=ps)

    # ---- Interface to the tests ----
    yield HERE / "trj.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "trj.npz"):
        os.remove(HERE / "trj.npz")


@pytest.fixture(scope="session")
def lti_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 32
    N = 51
    t_grid = np.linspace(0, 0.5, N)

    sampler = TrajectorySampler(
        f,
        g,
        config=HERE / "lti_data.yaml",
        rng=LTI_DATA_SEED,
        config_mod=config_chr,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(HERE / "lti.npz", t=ts, x=ys, u=us)

    # ---- Interface to the tests ----
    yield HERE / "lti.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "lti.npz"):
        os.remove(HERE / "lti.npz")


@pytest.fixture(scope="session")
def lti_gau():
    # ---- runs ONCE before any tests execute ----
    N = 51
    t_grid = np.linspace(0, 0.5, N)
    sampler = TrajectorySampler(
        f,
        g,
        config=HERE / "lti_data.yaml",
        rng=LTI_TEST_SEED,
        config_mod=config_gau,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    # ---- Interface to the tests ----
    yield (x_data, t_data, u_data)


@pytest.fixture(scope="session")
def kp_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 64
    N = 31
    t_grid = np.linspace(0, 0.5, N)

    sampler = TrajectorySampler(f_kp, config=HERE / "kp_data.yaml", rng=KP_DATA_SEED)
    ts, xs, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(HERE / "kp.npz", t=ts, x=ys)

    # ---- Interface to the tests ----
    yield HERE / "kp.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "kp.npz"):
        os.remove(HERE / "kp.npz")


@pytest.fixture(scope="session")
def kp_test():
    # ---- runs ONCE before any tests execute ----
    N = 31
    t_grid = np.linspace(0, 0.5, N)
    sampler = TrajectorySampler(
        f_kp,
        config=HERE / "kp_data.yaml",
        rng=KP_TEST_SEED,
        config_mod=config_gau,
    )
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    # ---- Interface to the tests ----
    yield (x_data, t_data)


@pytest.fixture(scope="session")
def ltg_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 32
    N = 51
    t_grid = np.linspace(0, 0.5, N)

    sampler = TrajectorySampler(
        f,
        g,
        config=HERE / "lti_data.yaml",
        rng=LTG_DATA_SEED,
        config_mod=config_chr,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)

    # Pretending a 3-node graph
    np.savez_compressed(
        HERE / "ltg.npz",
        t=ts,
        x=np.concatenate([ys, ys, ys], axis=-1),
        u=np.concatenate([us, us, us], axis=-1),
        p=np.concatenate([us[:, 0, :], us[:, 0, :], us[:, 0, :]], axis=-1).squeeze(),
        adj=adj,
    )

    # ---- Interface to the tests ----
    yield HERE / "ltg.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "ltg.npz"):
        os.remove(HERE / "ltg.npz")


@pytest.fixture(scope="session")
def ltg_gau():
    # ---- runs ONCE before any tests execute ----
    N = 51
    t_grid = np.linspace(0, 0.5, N)
    sampler = TrajectorySampler(
        f,
        g,
        config=HERE / "lti_data.yaml",
        rng=LTG_TEST_SEED,
        config_mod=config_gau,
    )
    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([ys[0], ys[0], ys[0]], axis=-1)
    t_data = ts[0]
    u_data = np.concatenate([us[0], us[0], us[0]], axis=-1)

    # ---- Interface to the tests ----
    yield (x_data, t_data, u_data, edge_index)


@pytest.fixture(scope="session")
def ltga_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 32
    N = 51
    t_grid = np.linspace(0, 0.5, N)

    sampler = TrajectorySampler(f_auto, g_auto, config=HERE / "ltga_data.yaml", rng=LTGA_DATA_SEED)
    ts, xs, ys = sampler.sample(t_grid, batch=B)

    # Pretending a 3-node graph
    np.savez_compressed(HERE / "ltga.npz", t=ts, x=np.concatenate([ys, ys, ys], axis=-1), adj=adj)

    # ---- Interface to the tests ----
    yield HERE / "ltga.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "ltga.npz"):
        os.remove(HERE / "ltga.npz")


@pytest.fixture(scope="session")
def ltga_test():
    N = 51
    t_grid = np.linspace(0, 0.5, N)
    sampler = TrajectorySampler(f_auto, config=HERE / "ltga_data.yaml", rng=LTGA_TEST_SEED)
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = np.concatenate([xs[0], xs[0], xs[0]], axis=-1)
    t_data = ts[0]
    edge_index = adj_to_edge(adj)[0]

    # ---- Interface to the tests ----
    yield (x_data, t_data, edge_index)


@pytest.fixture(scope="session")
def sa_lti_data():
    # ---- runs ONCE before any tests execute ----

    # --------------------
    # Data generation
    B = 64
    N = 21
    t_grid = np.linspace(0, 10, N)

    A = np.array([[0.0, 1.0], [-4.0, -1.0]])

    def f(t, x):
        return x @ A.T

    def g(t, x):
        return x

    sampler = TrajectorySampler(f, g, config=HERE / "sa_data.yaml", rng=SA_LTI_DATA_SEED)
    ts, xs, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(HERE / "sa.npz", t=ts, x=ys)

    # ---- Interface to the tests ----
    yield HERE / "sa.npz"

    # ---- runs ONCE after all tests finish (even on failure) ----

    # --------------------
    # Clean up
    if os.path.exists(HERE / "sa.npz"):
        os.remove(HERE / "sa.npz")


@pytest.fixture(scope="session")
def sa_lti_test():
    # ---- runs ONCE before any tests execute ----
    N = 21
    t_grid = np.linspace(0, 10, N)

    A = np.array([[0.0, 1.0], [-4.0, -1.0]])

    def f(t, x):
        return x @ A.T

    def g(t, x):
        return x

    sampler = TrajectorySampler(f, g, config=HERE / "sa_data.yaml", rng=SA_LTI_TEST_SEED)
    ts, xs, ys = sampler.sample(t_grid, batch=1)

    # ---- Interface to the tests ----
    yield (xs[0], ts[0])
