import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
import os

from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 128
N = 501
t_grid = np.linspace(0, 5, N)

def f(t, x, u):
    dxdt = np.zeros_like(x)
    k1 = 1.0
    k2 = 0.5
    b1 = 0.1
    b2 = 0.2
    dxdt[0] = x[1]
    dxdt[1] = -k1 * x[0] - b1 * x[1] + k2 * (x[2] - x[0]) + b2 * (x[3] - x[1]) + u 
    dxdt[2] = x[3]
    dxdt[3] = -k2 * (x[2] - x[0]) - b2 * (x[3] - x[1])
    return dxdt

g = lambda t, x, u: x

ifdat = 0
ifchk = 0
iftrn = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='spring_data.yaml')
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    os.makedirs('./data', exist_ok=True)
    np.savez_compressed('./data/spring.npz', t=ts, x=ys, u=us)

if ifchk:
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    data = np.load('./data/spring.npz')
    t = data['t']
    x = data['x']

    for ax_idx, ax in enumerate(axs.flat):
        idx = np.random.randint(0, x.shape[0])
        for i in range(x.shape[2]):
            ax.plot(t[idx], x[idx, :, i], label=f'x[{i}]')
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        ax.set_title(f'Trajectory #{idx}')
        ax.legend()
    plt.tight_layout()
    plt.show()

if iftrn:
    cases = {"model": LDM, "trainer": NODETrainer, "config": 'spring_ldm_node.yaml'}

    Model = cases['model']
    Trainer = cases['trainer']
    config = cases['config']

    setup_logging(config, prefix='results')
    logging.info(f"Config: {config}")
    trainer = Trainer(config, Model)
    trainer.train()