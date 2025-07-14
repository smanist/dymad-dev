import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


from dymad.models import LDM, KBF
from dymad.training import WeakFormTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

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

config_gau = {
    "control" : {
        "kind": "gaussian",
        "params": {
            "mean": 0.5,
            "std":  1.0,
            "t1":   4.0,
            "dt":   0.2,
            "mode": "zoh"}}}

dataproc = 1
iftrain=1


if dataproc:
    os.makedirs('./data', exist_ok=True)
    sampler = TrajectorySampler(f, g, config='ex1_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed('./data/ex1.npz', t=ts, x=ys, u=us)
    # print("yep")
if iftrain:
    cases={'model' : LDM, 'trainer': NODETrainer, 'config': 'ex1_ldm_node.yaml'}
    Model= cases['model']
    Trainer = cases['trainer']
    config_path = cases['config']
    setup_logging(config_path, mode='info', prefix='./logs')
    logging.info(f'Starting Training : {Model.__name__} with {Trainer.__name__} using config {config_path}')
    trainer= Trainer(config_path,Model)
    trainer.train()
    


