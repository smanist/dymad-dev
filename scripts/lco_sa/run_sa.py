# from pysako import resolventAnalysis, estimatePSpec

import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 400
N = 61
t_grid = np.linspace(0, 6, N)
dt = t_grid[1] - t_grid[0]

mu = 1.0
def f(t, x):
    _x, _y = x
    dx = np.array([
        _y,
        mu * (1-_x**2)*_y - _x
    ])
    return dx
g = lambda t, x: x

# Reference frequencies
_Nt, _T = 161, 40.0
_ts = np.linspace(0, _T, 8*_Nt)
_dt = _ts[1]
_res = spi.solve_ivp(f, [0,_T], [2,2], t_eval=_ts)
_tmp = _res.y[0,-4*_Nt:]
sp = np.fft.fft(_tmp)
fr = np.fft.fftfreq(4*_Nt)/_dt*(2*np.pi)
ii = np.argmax(np.abs(sp))
w0 = np.abs(fr[ii])
wa = np.exp(np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) * (1j*w0*dt))

mdl_kb = {
    "name" : 'sa_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "latent_dimension" : 32,
    "koopman_dimension" : 32,
    "activation" : "tanh",
    # "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}

mdl_kl = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 64,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [8, 8]}
    ]

trn_nd = {
    "n_epochs": 2000,
    "save_interval": 10,
    "load_checkpoint": False,
    "learning_rate": 1e-3,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 11, 21, 41],
    "sweep_epoch_step": 400,
    "chop_mode": "unfold",
    "chop_step": 0.5,
    "ls_update": {
        "method": "sako",
        "params": 9,
        "interval": 100,
        "times": 2,
        "start_with_ls": False}
        }
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "ls_update": {
        "method": "sako",
        "params": 0.2}
    }
config_path = 'sa_model.yaml'

cfgs = [
    ('kbf_nd',  KBF,  NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ]

IDX = [2]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod={'postprocess': {'n_skip': 20}})
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/sa.npz')

    for i in range(B):
        plt.plot(xs[i, :, 0], xs[i, :, 1])

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"sa_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

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
    # sadt = SpectralAnalysis(DKBF, 'sa_dkbf_nd.pt', dt=dt, reps=1e-10)
    sadt = SpectralAnalysis(DKBF, 'sa_dkbf_ln.pt', dt=dt, reps=1e-10)
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

            f, ax[0] = complex_plot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
            f, ax[0] = complex_plot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
            f, ax[1] = complex_plot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line')

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

            f, ax[0] = complex_plot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
            f, ax[0] = complex_plot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
            f, ax[1] = complex_plot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line')

        # for _i in range(2):
        #     ax[_i].set_xlim([-1.0, 0.5])
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
        plt.plot(th1, vg1, 'b-',  label='DT')
        plt.plot(th2, vg2, 'r--', label='CT')
        plt.plot([_arg[0], _arg[0]], [0, _amp], 'k:', label='System frequency')
        for _a in _arg[1:]:
            plt.plot([_a, _a], [0, _amp], 'k:')
        plt.legend()
        plt.xlabel('Angle, rad')
        plt.ylabel('Spectral measure')

    if ifegf:
        ## Eigenfunctions
        rngs = [[-np.pi/2.5, np.pi/2.5], [-1.4, 1.4]]
        Ns = [101, 121]
        md = 'abs'
        sadt.plot_eigfun_2d(rngs, Ns, 3, mode=md, ncols=3, figsize=(10,6))
        sadt.plot_eigfun_2d(rngs, Ns, 3, mode='angle', ncols=3, figsize=(10,6))
        sact.plot_eigfun_2d(rngs, Ns, 3, mode=md, ncols=3, figsize=(10,6))

plt.show()




"""
Limit Cycle Oscillations, which should consist of only point spectrum.

Rule of thumb:
1. All methods do not perform well in predicting transient responses.
2. ResDMD and K-ResDMD should perform similarly, with K-ResDMD slightly better in prediction and spectrum.
3. EDMD gives reasonable eigenfunctions regardless of data type; others do well for transient data (as trajs cover more space)

@author Dr. Daning Huang
@date 07/06/24
"""

