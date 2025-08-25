import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import load_model, plot_trajectory, setup_logging, TrajectorySampler

B = 64
N = 21
t_grid = np.linspace(0, 10, N)
dt = t_grid[1] - t_grid[0]

mu = -0.5
lm = -3
def f(t, x):
    _d = np.array([mu*x[0], lm*(x[1]-x[0]**2)])
    return _d

w0 = np.array([mu, lm]) + 1j*0
wa = np.exp(w0*dt)

mdl_kb = {
    "name" : 'kp_model',
    "encoder_layers" : 1,
    "decoder_layers" : 1,
    "latent_dimension" : 32,
    "koopman_dimension" : 8,
    "activation" : "tanh",
    "autoencoder_type" : "cat",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp"}
mdl_kl = {
    "name" : 'kp_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 9,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp"}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [3,3]}
    ]

trn_nd = {
    "n_epochs": 2000,
    "save_interval": 20,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "reconstruction_weight": 1.0,
    "dynamics_weight": 1.0,
    "sweep_lengths": [2, 4, 8, 21],
    "sweep_epoch_step": 400,
    "ls_update": {
        "method": "full",
        "interval": 200,
        "times": 2}
    }
trn_ln = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,
    "ls_update": {
        "method": "sako",
        "params": 4}
    }
config_path = 'kp_model.yaml'

cfgs = [
    ('dkbf_nd',   DKBF, NODETrainer,   {"model": mdl_kb, "training" : trn_nd}),
    ('kbf_nd',    KBF,  NODETrainer,   {"model": mdl_kb, "training" : trn_nd}),
    ('dkbf_ln',   DKBF, LinearTrainer, {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ]

# IDX = [0, 1]
# IDX = [2, 3]
IDX = [2]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifprd = 1
ifint = 0

if ifdat:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/kp.npz')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    for i in range(B):
        ax.plot(xs[i, :, 0], xs[i, :, 1], alpha=0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('2D Trajectories')
    plt.tight_layout()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for i in range(B):
        axs[0].plot(ts[i], xs[i, :, 0], alpha=0.5)
        axs[1].plot(ts[i], xs[i, :, 1], alpha=0.5)
    axs[0].set_ylabel('x1')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('x2')
    plt.tight_layout()

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"kp_{mdl}"
        setup_logging(config_path, mode='info', prefix='results')
        logging.info(f"Config: {config_path}")
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, config='kp_data.yaml')
    ts, xs, ys = sampler.sample(t_grid, batch=1)
    x_data = xs[0]
    t_data = ts[0]

    res = [x_data]
    for i in IDX:
        mdl, MDL, _, _ = cfgs[i]
        _, prd_func = load_model(MDL, f'kp_{mdl}.pt')
        with torch.no_grad():
            pred = prd_func(x_data, t_data)
        res.append(pred)

    plot_trajectory(
        np.array(res), t_data, "KP",
        labels=['Truth'] + labels, ifclose=False)

if ifint:
    sadt = SpectralAnalysis(DKBF, 'kp_dkbf_ln.pt', dt=dt, reps=1e-10)
    sact = SpectralAnalysis(KBF,  'kp_kbf_nd.pt',  dt=dt, reps=1e-10)

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





# mu = -0.5
# lm = -3

# # Sampling and data generation
# _T = np.eye(2)
# _S = np.eye(2)
# # ## Rotation
# # _t = np.pi/4
# # _c, _s = np.cos(_t), np.sin(_t)
# # _T = np.array([[_c, _s], [-_s, _c]])
# # _S = _T.T
# # ## Shear
# # _t = 0.2
# # _T = np.array([[1, _t], [0, 1]])
# # _S = np.array([[1, -_t], [0, 1]])
# def fNP(t, x):
#     _y = _T.dot(x)
#     _d = np.array([mu*_y[0], lm*(_y[1]-_y[0]**2)])
#     return _S.dot(_d)
# Jac = _S.dot(np.diag([mu, lm])).dot(_T)


# # Observables and the dataset
# K1, K2 = 3, 3
# fpsi = lambda x: psiCross(x, [psiMonomial, psiMonomial], [K1, K2])
# trans = [
#     ('lift', {
#         'fobs' : fpsi,
#         'finv' : [K2, 1]})]

# # Training and Filtering
# ## Regular EDMD, with filtering
# Nr = 4
# # edmd = EDMD(order='full', dt=dt)
# edmd = EDMD(order=Nr, dt=dt)
# edmd.fit(sol, trans=trans)

# ## ResDMD
# rdmd = ResDMD(order=edmd._Nrank, dt=dt, verbose=True)
# rdmd.fit(sol, trans=trans)
# sako = rdmd._sako

# ## Filtering for Regular EDMD
# edmd.filter_spectrum(sako)

# iftrj, ifprd, ifobs = 0, 0, 0
# ifeig, ifeic, ifpsp, ifres = 0, 0, 0, 0
# ifegf, iftrn = 0, 1

# prf = 'slm'
# ifsav = 1

# # Visualizations
# if iftrj:
#     ## Sample trajectories
#     f = plt.figure()
#     for _s in sol:
#         plt.plot(_s[:,0], _s[:,1])
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')
#     if ifsav:
#         f.savefig(f'./pics/{prf}_samples.png', dpi=600, bbox_inches='tight')

# if ifprd:
#     ## Predictions in original states
#     f, ax = rdmd.plot_pred_x(xt, ts, sts, idx='all')
#     if ifsav:
#         f.savefig(f'./pics/{prf}_predR.png', dpi=600, bbox_inches='tight')

#     f, ax = edmd.plot_pred_x(xt, ts, sts, idx='all')
#     if ifsav:
#         f.savefig(f'./pics/{prf}_predE.png', dpi=600, bbox_inches='tight')

# if ifobs:
#     ## Predictions in observables
#     Np = 3
#     idx = np.arange(Np)+1
#     f, ax = rdmd.plot_pred_psi(xt, ts, sts, idx=idx)
#     f, ax = edmd.plot_pred_psi(xt, ts, sts, idx=idx)

# if ifeig:
#     ## Eigenvalues
#     MRK = 15
#     f, ax = plt.subplots(ncols=2, sharey=True, figsize=(12,5))
#     f, ax[0], _ls = rdmd.plot_eigs(fig=(f, ax[0]))
#     ax[0].set_title(f'ResDMD, Max res: {rdmd._res[-1]:4.3e}')
#     ax[0].legend(_ls, ["Full-order", "Filtered"], loc=1)

#     f, ax[1], _l1 = edmd.plot_eigs(fig=(f, ax[1]))
#     ax[1].set_title(f'EDMD, Max res: {edmd._res[-1]:4.3e}')

#     if ifpsp:
#         # Pseudospectra
#         xs = np.linspace(-1.3,1.3,51)
#         gg = np.vstack([xs,xs])
#         rng = np.array([0.1, 0.25])
#         # ResDMD
#         grid, psrd = rdmd.estimate_ps(gg, mode='disc', method='sako', return_vec=False)
#         grid, psrs = rdmd.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
#         # EDMD
#         grid, psed = edmd.estimate_ps(gg, mode='disc', method='standard', return_vec=False)

#         f, ax[0] = cpPlot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
#         f, ax[0] = cpPlot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
#         f, ax[1] = cpPlot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line')

#     for _i in range(2):
#         ax[_i].set_xlim([-1.1, 1.1])
#         ax[_i].set_ylim([-1.1, 1.1])
#     if ifsav:
#         f.savefig(f'./pics/{prf}_cmp_eig.png', dpi=600, bbox_inches='tight')

# if ifeic:
#     ## Eigenvalues
#     MRK = 15
#     f, ax = plt.subplots(ncols=2, sharey=True, figsize=(12,5))
#     f, ax[0], _ls = rdmd.plot_eigs(fig=(f, ax[0]), mode='cont')
#     ax[0].set_title(f'ResDMD, Max res: {rdmd._res[-1]:4.3e}')
#     ax[0].legend(_ls, ["Full-order", "Filtered"], loc=1)

#     f, ax[1], _l1 = edmd.plot_eigs(fig=(f, ax[1]), mode='cont')
#     ax[1].set_title(f'EDMD, Max res: {edmd._res[-1]:4.3e}')

#     if ifpsp:
#         # Pseudospectra
#         zs = np.linspace(-1.2,1.2,51)
#         ws = np.linspace(-2.0,2.0,51)
#         gg = np.vstack([zs,ws])
#         rng = [0.04, 0.1, 0.25]
#         # ResDMD
#         grid, psrd = rdmd.estimate_ps(gg, mode='cont', method='sako', return_vec=False)
#         grid, psrs = rdmd.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
#         # EDMD
#         grid, psed = edmd.estimate_ps(gg, mode='cont', method='standard', return_vec=False)

#         f, ax[0] = cpPlot(grid, 1/psrd, rng, fig=(f, ax[0]), mode='line', lwid=1)
#         f, ax[0] = cpPlot(grid, 1/psrs, rng, fig=(f, ax[0]), mode='line', lsty='dotted')
#         f, ax[1] = cpPlot(grid, 1/psed, rng, fig=(f, ax[1]), mode='line')

#     for _i in range(2):
#         ax[_i].set_xlim([-1.5, 0.5])
#         ax[_i].set_ylim([-1.0, 1.0])
#     if ifsav:
#         f.savefig(f'./pics/{prf}_cmp_eic.png', dpi=600, bbox_inches='tight')

# if ifres:
#     ## Residuals
#     f, ax = plt.subplots()
#     ax.semilogy(np.abs(rdmd._wd_full), rdmd._res_full, 'bo', label='Full-order')
#     ax.semilogy(np.abs(rdmd._wd), rdmd._res, 'r^', label='ResDMD')
#     ax.semilogy(np.abs(edmd._wd), edmd._res, 'gs', markerfacecolor='none', label='EDMD')
#     ax.set_ylabel('Residual')
#     ax.set_xlabel('Norm of eigenvalue')
#     ax.legend()
#     if ifsav:
#         f.savefig(f'./pics/{prf}_cmp_res.png', dpi=600, bbox_inches='tight')

# if ifegf:
#     ## Eigenfunctions
#     rngs = [[-1,1],[-1,1]]
#     Ns = [101, 121]
#     rdmd.plot_eigfun_2d(rngs, Ns, 4, mode='real', ncols=2, figsize=(8,8))
#     if ifsav:
#         f.savefig(f'./pics/{prf}_eigfR.png', dpi=600, bbox_inches='tight')
#     # edmd.plot_eigfun_2d(rngs, Ns, 3, mode='real', ncols=2, figsize=(8,8))
#     # if ifsav:
#     #     f.savefig(f'./pics/{prf}_eigfE.png', dpi=600, bbox_inches='tight')

# if iftrn:
#     ## Conjugacy map
#     _Jw, _Jl, _Jr = scaledEig(Jac)
#     rdmd.set_conj_map(Jac)
#     ## Sample trajectories
#     stys = ['b-', 'r-', 'b--', 'r--']
#     f, ax = plt.subplots(ncols=3, figsize=(10,5))
#     for _i, _s in enumerate(sol):
#         ax[0].plot(_s[:,0], _s[:,1], stys[_i%4])
#         r0, r1 = rdmd.mapto_cnj(_s).real.T
#         ax[1].plot(r0, r1, stys[_i%4])
#         r0, r1 = rdmd.mapto_nrm(_s).real.T
#         ax[2].plot(r0, r1, stys[_i%4])
#     ## Slow manifold
#     yy = np.linspace(-1,1,41)
#     s0 = _S.dot(np.array([yy, yy**2])).T   # Closed-form solution is in transformed coord; so transform back
#     ax[0].plot(s0[:,0], s0[:,1], 'k-')
#     r0, r1 = rdmd.mapto_cnj(s0).real.T
#     ax[1].plot(r0, r1, 'k-')
#     r0, r1 = rdmd.mapto_nrm(s0).real.T
#     ax[2].plot(r0, r1, 'k-')
#     ## Linear basis
#     for _i in range(2):
#         for _j in range(2):
#             ax[_i].plot([0, _Jr[0,_j]], [0, _Jr[1,_j]], 'go-')
#     ax[2].plot([0, 0, 1], [1, 0, 0], 'go-')

#     ## Annotations
#     ax[0].set_xlabel(r'$x_1$')
#     ax[0].set_ylabel(r'$x_2$')
#     ax[0].set_title('Physical space')
#     # ax[0].set_aspect('equal')
#     ax[1].set_xlabel(r'$y_1$')
#     ax[1].set_ylabel(r'$y_2$')
#     ax[1].set_title('"Flatten" space')
#     # ax[1].set_aspect('equal')
#     ax[2].set_xlabel(r'$y_1^*$')
#     ax[2].set_ylabel(r'$y_2^*$')
#     ax[2].set_title('Orthogonalized space')
#     # ax[2].set_aspect('equal')
#     if ifsav:
#         f.savefig(f'./pics/{prf}_trans.png', dpi=600, bbox_inches='tight')

# plt.show()