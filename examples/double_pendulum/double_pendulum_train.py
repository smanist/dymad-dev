import logging
import os

import numpy as np

from dymad.models import KBF, LDM
from dymad.training import NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, setup_logging

logging.basicConfig(level=logging.INFO)

B = 128
N = 501
DATA_PATH = "./data/dp_data.npz"
t_grid = np.linspace(0, 5, N)


def f(t, x, u):
    dx = np.zeros_like(x)
    # Physical parameters
    m1, m2 = 1.0, 1.0  # masses
    l1, l2 = 1.0, 1.0  # lengths
    g = 9.81  # gravity
    th1, th2, w1, w2 = x[0], x[1], x[2], x[3]

    dx[0] = w1
    dx[1] = w2
    dx[2] = (
        -g * (2 * m1 + m2) * np.sin(th1)
        - m2 * g * np.sin(th1 - 2 * th2)
        - 2 * np.sin(th1 - th2) * m2 * (l2 * w2**2 + l1 * w1**2 * np.cos(th1 - th2))
    ) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)))
    dx[3] = (
        2
        * np.sin(th1 - th2)
        * (
            l1 * w1**2 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1)
            + l2 * w2**2 * m2 * np.cos(th1 - th2)
        )
    ) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)))
    return dx


def g(t, x, u):
    return x


def describe_model(model):
    model_spec = getattr(model, "model_spec", None)
    if model_spec is not None and getattr(model_spec, "name", None):
        return model_spec.name
    return getattr(model, "__name__", model.__class__.__name__)


def ensure_dataset(path=DATA_PATH):
    if os.path.exists(path):
        return
    logging.info("Generating double pendulum dataset at %s", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sampler = TrajectorySampler(f, g, config="double_pendulum_data.yaml")
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed(path, t=ts, x=ys, u=us)


ifdat = 0  # Generate data
ifchk = 0  # Check Data
iftrn = 1  # Train model

if ifdat:
    ensure_dataset()

if ifchk:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    ensure_dataset()
    data = np.load(DATA_PATH)
    t = data["t"]
    x = data["x"]

    for _ax_idx, ax in enumerate(axs.flat):
        idx = np.random.randint(0, x.shape[0])
        for i in range(x.shape[2]):
            ax.plot(t[idx], x[idx, :, i], label=f"x[{i}]")
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title(f"Trajectory #{idx}")
        ax.legend()
    plt.tight_layout()
    plt.show()

case = 0
if iftrn:
    ensure_dataset()
    cases = [
        {"model": LDM, "trainer": NODETrainer, "config": "dp_ldm_node.yaml"},
        {"model": LDM, "trainer": WeakFormTrainer, "config": "dp_ldm_wf.yaml"},
        {"model": KBF, "trainer": NODETrainer, "config": "dp_kbf_node.yaml"},
        {"model": KBF, "trainer": WeakFormTrainer, "config": "dp_kbf_wf.yaml"},
    ]

    Model = cases[case]["model"]
    Trainer = cases[case]["trainer"]
    config_path = cases[case]["config"]
    setup_logging(config_path, mode="info", prefix="./logs")
    logging.info(
        "Starting Training : %s with %s using config %s",
        describe_model(Model),
        Trainer.__name__,
        config_path,
    )
    trainer = Trainer(config_path, Model)
    trainer.train()
