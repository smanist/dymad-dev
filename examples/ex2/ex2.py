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

ifdata = 0
iftrain=0
ifplot = 1


if ifdata:
    os.makedirs('./data', exist_ok=True)
    sampler = TrajectorySampler(f, g, config='ex2_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed('./data/ex2.npz', t=ts, x=ys, u=us)
    
if iftrain:
    cases=[{'model' : LDM, 'trainer': WeakFormTrainer, 'config': 'ex2_ldm_wf.yaml'},
           {'model' : LDM, 'trainer': NODETrainer, 'config': 'ex2_ldm_node.yaml'}]
    

    Model= cases[0]['model']
    Trainer = cases[0]['trainer']
    config_path = cases[0]['config']
    setup_logging(config_path, mode='info', prefix='./logs')
    logging.info(f'Starting Training : {Model.__name__} with {Trainer.__name__} using config {config_path}')
    trainer= Trainer(config_path,Model)
    trainer.train()

    Model= cases[1]['model']
    Trainer = cases[1]['trainer']
    config_path = cases[1]['config']
    setup_logging(config_path, mode='info', prefix='./logs')
    logging.info(f'Starting Training : {Model.__name__} with {Trainer.__name__} using config {config_path}')
    trainer= Trainer(config_path,Model)
    trainer.train()

if ifplot:
    sumloss = np.load(f'results/lti_ldm_summary.npz')
    eloss, hloss= sumloss['epoch_loss'], sumloss['losses']
    ermse, hrmse = sumloss['epoch_rmse'], sumloss['rmses']

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    ax[0].semilogy(eloss, hloss[0]/hloss[0][0])

    ax[1].semilogy(ermse, hrmse[0],'g-', label='Train')
    ax[1].semilogy(ermse, hrmse[1],'r-', label='Validation')
    ax[1].semilogy(ermse, hrmse[2],'b-', label='Test')

    # Plotting code here
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('Epoch')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    


