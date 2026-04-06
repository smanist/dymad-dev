from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dymad.utils import TrajectorySampler

HERE = Path(__file__).parent

config_path = HERE / "lti_data.yaml"
config_para = HERE / "lti_data_par.yaml"

B = 32
N = 101
t_grid = np.linspace(0, 5, N)

A = np.array([[0.0, 1.0], [-1.0, -0.1]])


def f(t, x, u):
    return (x @ A.T) + u


def fp(t, x, u, p):
    return p * (x @ A.T) + u


def g(t, x, u):
    return x


ctrl_sin = {
    "kind": "sine",
    "params": {
        "num_components": 2,
        "freq_range": (0.5, 2.0),
        "amp_range": (0.2, 1.0),
        "phase_range": (0, 360),
    },
}
ctrl_gau = {
    "kind": "gaussian",
    "params": {"mean": 0.5, "std": 1.0, "t1": 4.0, "dt": 0.2, "mode": "zoh"},
}
ctrl_chr = {
    "kind": "chirp",
    "params": {
        "t1": 4.0,
        "freq_range": (0.5, 2.0),
        "amp_range": (0.5, 1.0),
        "phase_range": (0.0, 360.0),
    },
}
ctrl_sph = {"kind": "sphere", "params": {"radius": 0.5, "t1": 4.0, "dt": 0.2, "mode": "zoh"}}

x0_ga1 = {"kind": "gaussian", "params": {"mean": 0.0, "std": 1.0}}
x0_ga2 = {"kind": "gaussian", "params": {"mean": [0.0, 1.0], "std": [2.0, 1.0]}}
x0_uni = {"kind": "uniform", "params": {"bounds": [[-1.0, 1.0], [0.0, 1.0]]}}
x0_grd = {"kind": "grid", "params": {"bounds": [[-1.0, 1.0], [0.0, 1.0]], "n_points": [15, 10]}}
x0_per = {
    "kind": "perturb",
    "params": {"bounds": [[-1.0, 1.0], [0.0, 1.0]], "ref": np.array([[3, 3], [3, -3], [-3, -3]])},
}

p_uni = {"kind": "uniform", "params": {"bounds": [[0.0, 1.0]]}}

ctrls = [ctrl_sin, ctrl_gau, ctrl_chr, ctrl_sph, ctrl_chr, ctrl_chr]
x0s = [x0_ga1, x0_ga2, x0_uni, x0_grd, x0_per, x0_per]
ps = [None, None, None, None, None, p_uni]
ttls = [
    "Sine+Gaussian1",
    "Gaussian+Gaussian2",
    "Chirp+Uniform",
    "Sphere+Grid",
    "Perturbation",
    "Perturbation/Param",
]


def sample_case(j):
    config_mod = {"control": ctrls[j], "x0": x0s[j], "p": ps[j]}
    if ps[j] is None:
        sampler = TrajectorySampler(f, g, config=config_path, config_mod=config_mod)
    else:
        sampler = TrajectorySampler(fp, g, config=config_para, config_mod=config_mod)
    return sampler.sample(t_grid, batch=B)


def plot_samples(ts, xs, us, j):
    x0 = xs[:, 0, :].squeeze().T

    fig, ax = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 8), sharex=True, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left column: trajectories
    for i in range(B):
        ax[0, 0].plot(ts[i], xs[i, :, 0], "b-", alpha=0.1)
        ax[1, 0].plot(ts[i], xs[i, :, 1], "b-", alpha=0.1)
        ax[2, 0].plot(ts[i], us[i], "b-", alpha=0.1)
    ax[0, 0].set_title(ttls[j])
    ax[0, 0].set_ylabel("x1")
    ax[1, 0].set_ylabel("x2")
    ax[2, 0].set_ylabel("u")
    ax[2, 0].set_xlabel("t")

    # Right column: initial conditions
    gs = ax[0, 0].get_gridspec()
    for a in ax[:, 1]:
        a.remove()
    ax_ic = fig.add_subplot(gs[:, 1])
    ax_ic.plot(x0[0], x0[1], "bo", markerfacecolor="none", alpha=0.5)
    ax_ic.set_xlabel("x1")
    ax_ic.set_ylabel("x2")
    ax_ic.set_title("Initial conditions")
    ax_ic.set_aspect("equal", adjustable="box")


@pytest.mark.parametrize("idx", range(len(ctrls)))
def test_sampling(idx, plot=False):
    res = sample_case(idx)

    if plot:
        ts, xs, us = res[:3]
        plot_samples(ts, xs, us, idx)
        plt.tight_layout()

        if len(res) == 5:
            plt.figure()
            plt.hist(res[4].flatten(), bins=5)


if __name__ == "__main__":
    for j in range(len(ctrls)):
        test_sampling(j, plot=True)

    plt.show()
