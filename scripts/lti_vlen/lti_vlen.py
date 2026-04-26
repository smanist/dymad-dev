import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dymad.models import LDM
from dymad.training import NODETrainer, WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_summary

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Keep a shared sampling interval across trajectories. The weak-form path currently
# builds one set of integration weights from dataset metadata, so this example
# varies trajectory length only and leaves mixed-dt support explicit for follow-up work.
DT = 0.05
LENGTHS = [41, 61, 81, 101]
TRAJ_PER_LENGTH = 8

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


def _to_object_array(items):
    data = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        data[index] = np.asarray(item)
    return data


def generate_variable_length_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    sampler = TrajectorySampler(f, g, config=BASE_DIR / "lti_vlen_data.yaml", rng=7)

    ts_all = []
    ys_all = []
    us_all = []
    for n_steps in LENGTHS:
        t_grid = DT * np.arange(n_steps)
        ts, _, us, ys = sampler.sample(t_grid, batch=TRAJ_PER_LENGTH)
        ts_all.extend(np.array(item) for item in ts)
        ys_all.extend(np.array(item) for item in ys)
        us_all.extend(np.array(item) for item in us)

    # Store ragged object arrays on purpose; padding here would defeat the example.
    np.savez_compressed(
        DATA_DIR / "lti_vlen.npz",
        t=_to_object_array(ts_all),
        x=_to_object_array(ys_all),
        u=_to_object_array(us_all),
    )


def train_node() -> None:
    trainer = NODETrainer(str(BASE_DIR / "lti_vlen_ldm_node.yaml"), LDM)
    trainer.train()


def train_weak_form() -> None:
    trainer = WeakFormTrainer(str(BASE_DIR / "lti_vlen_ldm_wf.yaml"), LDM)
    trainer.train()


ifdat = 1
ifnode = 1
ifwf = 1
ifplt = 1


def main() -> None:
    os.chdir(BASE_DIR)

    if ifdat:
        generate_variable_length_data()

    if ifnode:
        train_node()

    if ifwf:
        train_weak_form()

    if ifplt:
        plot_summary(
            ["lti_vlen_ldm_node", "lti_vlen_ldm_wf"],
            labels=["NODE", "Weak form"],
            ifclose=False,
        )

    plt.show()


if __name__ == "__main__":
    main()
