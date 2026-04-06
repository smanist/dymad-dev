import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

from dymad.io import load_model
from dymad.models import KBF
from dymad.training import WeakFormTrainer
from dymad.utils import TrajectorySampler, plot_cv_results, plot_multi_trajs

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def g(t, x, u):
    return x


config_gau = {
    "control": {
        "kind": "gaussian",
        "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
    }
}

cases = [
    {"name": "kbf_cv", "model": KBF, "trainer": WeakFormTrainer, "config": "lti_kbf_cv.yaml"},
]
IDX = [0]
labels = [cases[i]["name"] for i in IDX]

if __name__ == "__main__":
    iftrn = 1
    ifplt = 1
    ifprd = 1

    if iftrn:
        mp.set_start_method("spawn", force=True)
        for _i in IDX:
            Model = cases[_i]["model"]
            Trainer = cases[_i]["trainer"]
            config_path = cases[_i]["config"]

            trainer = Trainer(config_path, Model, max_workers=4)
            trainer.train()

    if ifplt:
        mdl = cases[0]["name"]
        keys = ["model.koopman_dimension", "training.weak_form_params.N"]
        # keys = ['model.koopman_dimension']
        # keys = None
        plot_cv_results(f"lti_{mdl}", keys, ifclose=False)

    if ifprd:
        sampler = TrajectorySampler(f, g, config="lti_data.yaml", config_mod=config_gau)
        ts, xs, us, ys = sampler.sample(t_grid, batch=3)

        res = [xs]
        for _i in IDX:
            mdl, MDL = cases[_i]["name"], cases[_i]["model"]
            _, prd_func = load_model(MDL, f"lti_{mdl}.pt")

            with torch.no_grad():
                _pred = prd_func(xs, ts, u=us)
            res.append(_pred)

        plot_multi_trajs(
            np.array(res), ts[0], "LTI", us=us, labels=["Truth"] + labels, ifclose=False
        )

    plt.show()
