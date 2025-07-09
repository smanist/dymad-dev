import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

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

ifdat = 0
iftrn = 1
ifplt = 1
ifprd = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_chr)
    ts, xs, us, ys = sampler.sample(t_grid, batch=B)
    np.savez_compressed('./data/lti.npz', t=ts, x=ys, u=us)

if iftrn:
    cases = [
        {"model" : LDM, "trainer": WeakFormTrainer, "config": 'lti_ldm_wf.yaml'},
        {"model" : LDM, "trainer": NODETrainer,     "config": 'lti_ldm_node.yaml'},
        {"model" : KBF, "trainer": WeakFormTrainer, "config": 'lti_kbf_wf.yaml'},
        {"model" : KBF, "trainer": NODETrainer,     "config": 'lti_kbf_node.yaml'}
    ]

    for _i in [0, 1]:
        Model = cases[_i]['model']
        Trainer = cases[_i]['trainer']
        config_path = cases[_i]['config']

        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, Model)
        trainer.train()

if ifplt:
    mdl = 'ldm'
    # mdl = 'kbf'
    sum_wf = np.load(f'results/lti_{mdl}_wf_summary.npz')
    sum_nd = np.load(f'results/lti_{mdl}_node_summary.npz')

    e_loss_wf, h_loss_wf = sum_wf['epoch_loss'], sum_wf['losses']
    e_loss_nd, h_loss_nd = sum_nd['epoch_loss'], sum_nd['losses']
    e_rmse_wf, h_rmse_wf = sum_wf['epoch_rmse'], sum_wf['rmses']
    e_rmse_nd, h_rmse_nd = sum_nd['epoch_rmse'], sum_nd['rmses']

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))
    ax[0].semilogy(e_loss_wf, h_loss_wf[0]/h_loss_wf[0,0], 'r-', label='Weak Form')
    ax[0].semilogy(e_loss_nd, h_loss_nd[0]/h_loss_nd[0,0], 'b-', label='NODE')
    ax[0].set_title('Training Loss (relative)')
    ax[0].set_ylabel('Relative Loss')
    ax[0].legend()

    ax[1].semilogy(e_rmse_wf, h_rmse_wf[0], 'r-', label='Weak Form, Train')
    ax[1].semilogy(e_rmse_wf, h_rmse_wf[2], 'r--', label='Weak Form, Test')
    ax[1].semilogy(e_rmse_nd, h_rmse_nd[0], 'b-', label='NODE, Train')
    ax[1].semilogy(e_rmse_nd, h_rmse_nd[2], 'b--', label='NODE, Test')
    ax[1].set_title('Traj RMSE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('RMSE')
    ax[1].legend()

    print("Epoch time NODE/WF:", sum_nd['avg_epoch_time']/sum_wf['avg_epoch_time'])

if ifprd:
    MDL, mdl = LDM, 'ldm'
    # MDL, mdl = KBF, 'kbf'
    mdl_wf, prd_wf = load_model(MDL, f'lti_{mdl}_wf.pt', f'lti_{mdl}_wf.yaml')
    mdl_nd, prd_nd = load_model(MDL, f'lti_{mdl}_node.pt', f'lti_{mdl}_node.yaml')

    sampler = TrajectorySampler(f, g, config='lti_data.yaml', config_mod=config_gau)

    ts, xs, us, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]
    u_data = us[0]

    with torch.no_grad():
        weak_pred = prd_wf(x_data, u_data, t_data)
        node_pred = prd_nd(x_data, u_data, t_data)

    plot_trajectory(
        np.array([x_data, weak_pred, node_pred]), t_data, "LTI", metadata={'n_state_features': 2},
        us=u_data, labels=['Truth', 'Weak Form', 'NODE'], ifclose=False)

plt.show()
