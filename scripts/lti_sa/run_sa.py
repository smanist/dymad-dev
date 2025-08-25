# from pysako import resolventAnalysis, estimatePSpec

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spl
import torch

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import estimate_pseudospectrum, resolvent_analysis, SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 64
N = 21
t_grid = np.linspace(0, 10, N)
dt = t_grid[1] - t_grid[0]

# prf = 'lti_har'
# A = np.array([
#     [0.0, 1.0],
#     [-4.0, 0.0]])
prf = 'lti_dmp'
A = np.array([
    [0.0, 1.0],
    [-4.0, -1.0]])
# prf = 'lti_dgn'
# A = 0.5*np.array([
#     [-1.0,-0.9],
#     [0.0, -1.0]])

def f(t, x):
    return (x @ A.T)
g = lambda t, x: x

# True eigenvalues
wa = np.exp(np.linalg.eig(A)[0]*dt)
w0 = np.linalg.eig(A)[0]

mdl_kb = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 2,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",
    }

mdl_kl = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 4,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [2, 2]}
    ]

trn_nd = {
    "n_epochs": 300,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4],
    "sweep_epoch_step": 100,
    "ode_method": "dopri5",
    "ode_args": {
        "rtol": 1e-7,
        "atol": 1e-9},
    "ls_update": {
        "method": "full",
        "interval": 50,
        "times": 1}
        }
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "ls_update": {
        "method": "sako",
        "params": 2}
    }
config_path = 'sa_model.yaml'

cfgs = [
    ('kbf_nd',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ]

# IDX = [0, 1]
IDX = [2]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 0
ifprd = 0
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/sa.npz')

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"sa_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, config='sa_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'sa_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "SA",
        labels=['Truth'] + labels, ifclose=False)

if ifint:
    sadt = SpectralAnalysis(DKBF, 'sa_dkbf_nd.pt', dt=dt, reps=1e-10)
    sact = SpectralAnalysis(KBF,  'sa_kbf_nd.pt',  dt=dt, reps=1e-10)

    ifeig, ifeic, ifpsp, ifres = 1, 1, 1, 1
    ifspe, ifegf = 1, 1

    if ifeig:
        ## Eigenvalues
        MRK = 15
        f, ax = plt.subplots(ncols=2, sharey=True, figsize=(12,5))
        f, ax[0], _ls = sadt.plot_eigs(fig=(f, ax[0]))
        _l, = ax[0].plot(wa.real, wa.imag, 'kx', markersize=MRK)
        ax[0].set_title(f'DT, Max res: {sadt._res[-1]:4.3e}')
        ax[0].legend(_ls+[_l], ["Full-order", "Filtered", "Truth"], loc=1)

        f, ax[1], _l1 = sact.plot_eigs(fig=(f, ax[1]))
        _l, = ax[1].plot(wa.real, wa.imag, 'kx', markersize=MRK)
        ax[1].set_title(f'CT, Max res: {sact._res[-1]:4.3e}')

        if ifpsp:
            # Pseudospectra
            xs = np.linspace(-1.3, 1.3, 51)
            gg = np.vstack([xs, xs])
            rng = np.array([0.1, 0.25])
            # DT
            grid, psrd = sadt.estimate_ps(gg, mode='disc', method='sako', return_vec=False)
            grid, psrs = sadt.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
            # CT
            grid, psed = sact.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
            # Exact
            psrf = estimate_pseudospectrum(
                grid, resolvent_analysis, return_vec=False,
                A=spl.expm(A*dt), B=None, ord=1)

            f, ax[0] = complex_plot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
            f, ax[0] = complex_plot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
            f, ax[0] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[0]), mode='line', lwid=1, lsty='dashed')
            f, ax[1] = complex_plot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line', lsty='dotted')
            f, ax[1] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[1]), mode='line', lwid=1, lsty='dashed')

        for _i in range(2):
            ax[_i].set_xlim([-0.1, 1.3])
            ax[_i].set_ylim([-1.1, 1.1])

    if ifeic:
        ## Eigenvalues
        MRK = 15
        f, ax = plt.subplots(ncols=2, sharey=True, figsize=(12,5))
        f, ax[0], _ls = sadt.plot_eigs(fig=(f, ax[0]), mode='cont')
        _l, = ax[0].plot(w0.real, w0.imag, 'kx', markersize=MRK)
        ax[0].set_title(f'DT, Max res: {sadt._res[-1]:4.3e}')
        ax[0].legend(_ls+[_l], ["Full-order", "Filtered", "Truth"], loc=1)

        f, ax[1], _l1 = sact.plot_eigs(fig=(f, ax[1]), mode='cont')
        _l, = ax[1].plot(w0.real, w0.imag, 'kx', markersize=MRK)
        ax[1].set_title(f'CT, Max res: {sact._res[-1]:4.3e}')

        if ifpsp:
            # Pseudospectra
            zs = np.linspace(-1.0,0.5,51)
            ws = np.linspace(-3.0,3.0,51)
            gg = np.vstack([zs,ws])
            rng = np.array([0.1, 0.25])
            # DT
            grid, psrd = sadt.estimate_ps(gg, mode='cont', method='sako', return_vec=False)
            grid, psrs = sadt.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
            # CT
            grid, psed = sact.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
            # Exact
            psrf = estimate_pseudospectrum(
                grid, resolvent_analysis, return_vec=False,
                A=A, B=None, ord=1)

            f, ax[0] = complex_plot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
            f, ax[0] = complex_plot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
            f, ax[0] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[0]), mode='line', lwid=1, lsty='dashed')
            f, ax[1] = complex_plot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line', lsty='dotted')
            f, ax[1] = complex_plot(grid, 1/psrf, rng, fig=(f, ax[1]), mode='line', lwid=1, lsty='dashed')

        for _i in range(2):
            ax[_i].set_xlim([-1.0, 0.5])
#         ax[_i].set_ylim([-3.0, 3.0])

    if ifres:
        ## Residuals
        f, ax = plt.subplots()
        ax.semilogy(np.abs(sadt._wd_full), sadt._res_full, 'bo', label='Full-order')
        ax.semilogy(np.abs(sadt._wd), sadt._res, 'r^', label='DT')
        ax.semilogy(np.abs(sact._wd), sact._res, 'gs', markerfacecolor='none', label='CT')
        ax.set_xlabel('Norm of eigenvalue')
        ax.set_ylabel('Residual')
        ax.legend()

    if ifspe:
        # Spectral measure
        def func_obs(x):
            _x1, _x2 = x.T
            return _x1+_x2
        gobs = sact.apply_obs(func_obs)
        th1, vg1 = sadt._sako.estimate_measure(gobs, 6, 0.1, thetas=501)
        th2, vg2 = sact._sako.estimate_measure(gobs, 6, 0.1, thetas=501)

        _arg = np.angle(wa)
        _amp = np.max(vg1)

        f = plt.figure()
        plt.plot(th1, vg1, 'b-', label='DT')
        plt.plot(th2, vg2, 'r-', label='CT')
        plt.plot([_arg[0], _arg[0]], [0, _amp], 'k:', label='System frequency')
        plt.plot([_arg[1], _arg[1]], [0, _amp], 'k:')
        plt.legend()
        plt.xlabel('Angle, rad')
        plt.ylabel('Spectral measure')

    if ifegf:
        ## Eigenfunctions
        rngs = [[-np.pi/2.5, np.pi/2.5], [-1.4, 1.4]]
        Ns = [101, 121]
        md = 'real'
        sadt.plot_eigfun_2d(rngs, Ns, 2, mode=md, ncols=2, figsize=(10,6))
        sact.plot_eigfun_2d(rngs, Ns, 2, mode=md, ncols=2, figsize=(10,6))

plt.show()