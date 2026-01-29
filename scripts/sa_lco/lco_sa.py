import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
import torch

from dymad.io import load_model
from dymad.models import DKBF, KBF
from dymad.numerics import complex_plot
from dymad.sako import per_state_err, SpectralAnalysis
from dymad.training import LinearTrainer, NODETrainer
from dymad.utils import plot_trajectory, TrajectorySampler

B = 500
N = 81
t_grid = np.linspace(0, 8, N)
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

# Reference trajectory
_Nt = 161
_ts = np.linspace(0, 40.0, 8*_Nt)
_res = spi.solve_ivp(f, [0,_ts[-1]], [2,2], t_eval=_ts)
_ref = _res.y[:,-240:].T

# Reference frequencies
_tmp = _res.y[0,-4*_Nt:]
_dt = _ts[1]           # dt from reference
sp = np.fft.fft(_tmp)
fr = np.fft.fftfreq(4*_Nt)/_dt*(2*np.pi)
ii = np.argmax(np.abs(sp))
w0 = np.abs(fr[ii])
wc = np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) * (1j*w0)
wa = np.exp(wc*dt) # Use dt from data

# Transition to LCO
db = 0.4

mdl_kb = {
    "name" : 'sa_model',
    "encoder_layers" : 2,
    "decoder_layers" : 2,
    "hidden_dimension" : 64,
    "koopman_dimension" : 3,
    "activation" : "prelu",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}

mdl_kl = {
    "name" : 'sa_model',
    "encoder_layers" : 0,
    "decoder_layers" : 0,
    "koopman_dimension" : 100,
    "activation" : "none",
    "weight_init" : "xavier_uniform",
    "predictor_type" : "exp",}
trn_kl = [
        {"type": "scaler", "mode": "-11"},
        {"type": "lift", "fobs": "poly", "Ks": [10, 10]}
    ]

crit = {
    "dynamics": {"weight": 1.0},
    "recon":    {"weight": 1.0},
}
trn_dt = {
    "n_epochs": 5000,
    "save_interval": 100,
    "load_checkpoint": False,
    "learning_rate": 1e-2,
    "decay_rate": 0.999,
    "ls_update": {
        "method": "full",
        "interval": 500,
        "times": 1}
        }
trn_ct = copy.deepcopy(trn_dt)
trn_ct["ls_update"]["method"] = "full_log"

ref = {
    "n_epochs": 1,
    "save_interval": 1,
    "load_checkpoint": False,}
trn_ln = {
    "ls_update": {
        "method": "full"}}
trn_ln.update(ref)
trn_tr = {
    "ls_update": {
        "method": "truncated",
        "params": 15}}
trn_tr.update(ref)
trn_sa = {
    "ls_update": {
        "method": "sako",
        "params": 15,
        "remove_one": True}}
trn_sa.update(ref)

smpl = {'x0': {
    'kind': 'perturb',
    'params': {'bounds': [-db, db], 'ref': _ref}}
    }
config_path = 'sa_model.yaml'

cfgs = [
    ('kbf_nd',  KBF,  NODETrainer,     {"model": mdl_kb, "criterion": crit, "training" : trn_ct}),
    ('dkbf_nd', DKBF, NODETrainer,     {"model": mdl_kb, "criterion": crit, "training" : trn_dt}),
    ('dkbf_ln', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_ln}),
    ('dkbf_tr', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_tr}),
    ('dkbf_sa', DKBF, LinearTrainer,   {"model": mdl_kl, "transform_x" : trn_kl, "training" : trn_sa}),
    ]

IDX = [0, 1, 2, 3, 4]
# IDX = [0, 1]
# IDX = [2, 3, 4]
labels = [cfgs[i][0] for i in IDX]

ifdat = 0
iftrn = 1
ifprd = 1
ifint = 1

if ifdat:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
    ts, xs, ys = sampler.sample(t_grid, batch=B, save='./data/sa.npz')

    for i in range(B):
        plt.plot(ys[i, :, 0], ys[i, :, 1])
    plt.plot(_ref[:,0], _ref[:,1], 'k--', linewidth=2)

if iftrn:
    for i in IDX:
        mdl, MDL, Trainer, opt = cfgs[i]
        opt["model"]["name"] = f"sa_{mdl}"
        trainer = Trainer(config_path, MDL, config_mod=opt)
        trainer.train()

if ifprd:
    sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
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
    saln = SpectralAnalysis(DKBF, 'sa_dkbf_ln.pt', dt=dt, reps=1e-10, etol=None)
    satr = SpectralAnalysis(DKBF, 'sa_dkbf_tr.pt', dt=dt, reps=1e-10, etol=None)
    sasa = SpectralAnalysis(DKBF, 'sa_dkbf_sa.pt', dt=dt, reps=1e-10, etol=None)
    sadt = SpectralAnalysis(DKBF, 'sa_dkbf_nd.pt', dt=dt, reps=1e-10)
    sact = SpectralAnalysis(KBF,  'sa_kbf_nd.pt',  dt=dt, reps=1e-10)

    sas = [saln, satr, sasa, sadt, sact]
    lbs = ['DT-LN', 'DT-TR', 'DT-SA', 'DT-ND', 'CT-ND']

    Ns  = len(sas)

    ifprd, ifcnv = 1, 1
    ifeig, ifeic, ifpsp, ifres = 1, 1, 1, 0
    ifspe, ifegf = 1, 1

    if ifprd:
        J = 32
        sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
        ts, xs, _ = sampler.sample(t_grid, batch=J)
        x0s = xs[:, 0, :].squeeze()

        fig, ax = plt.subplots(nrows=2, ncols=Ns, sharex=True, sharey=True, figsize=(12,6))
        for _i in range(Ns):
            sas[_i].plot_pred(x0s, ts[0], ref=xs, title=lbs[_i], fig=(fig, ax[:,_i]))

    if ifcnv:
        J = 32
        sampler = TrajectorySampler(f, g, config='sa_data.yaml', config_mod=smpl)
        ts, xs, _ = sampler.sample(t_grid, batch=J)
        x0s = xs[:, 0, :].squeeze()

        errs = []
        # for _o in [1.0, 0.66, 0.45, 0.27, 0.19]:
        for _o in [3, 9, 15, 27, 47, 65]:
            saln.filter_spectrum(order=_o, remove_one=False)
            prd = saln.predict(x0s, ts[0])
            err = per_state_err(prd.real, xs)
            errs.append([saln._Nrank, np.mean(err)])
            # saln.plot_pred(x0s, ts[0], ref=xs, idx='all', figsize=(6,8), title=f"Order {saln._Nrank}")
        errs = np.array(errs).T

        prd = sasa.predict(x0s, ts[0])
        esa = per_state_err(prd.real, xs)
        # sasa.plot_pred(x0s, ts[0], ref=xs, idx='all', figsize=(6,8), title='DT-SA')

        prd = satr.predict(x0s, ts[0])
        etr = per_state_err(prd.real, xs)

        fig = plt.figure()
        plt.plot(errs[0], errs[1], 'bo-', label='DT-LN', markerfacecolor='none')
        plt.plot(sasa._Nrank, np.mean(esa), 'rs', label='DT-SA')
        plt.plot(satr._Nrank, np.mean(etr), 'g^', label='DT-TR')
        plt.legend()
        plt.xlabel('Order')
        plt.ylabel('Error')

    if ifeig:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=Ns, sharey=True, figsize=(12,5))
        for _i in range(Ns):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]), plot_filt=None)
            _l, = ax[_i].plot(wa.real, wa.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], ["Full-Order", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            xs = np.linspace(-1.3, 1.3, 51)
            gg = np.vstack([xs, xs])
            rng = np.array([0.1, 0.25])

            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='disc', method='standard', return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode='disc', method='sako', return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(Ns):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                f, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(f, ax[_i]), mode='line', lwid=2, lsty='dotted')
                f, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(f, ax[_i]), mode='line', lwid=1)

        for _i in range(Ns):
            ax[_i].set_xlim([-0.1, 1.3])
            ax[_i].set_ylim([-1.1, 1.1])

    if ifeic:
        ## Eigenvalues
        MRK = 15
        fig, ax = plt.subplots(ncols=Ns, sharey=True, figsize=(15,5))
        for _i in range(Ns):
            fig, ax[_i], _ls = sas[_i].plot_eigs(fig=(fig, ax[_i]), mode='cont', plot_filt=None)
            _l, = ax[_i].plot(wc.real, wc.imag, 'kx', markersize=MRK)
            ax[_i].set_title(f'{lbs[_i]}\nMax res: {sas[_i]._res[-1]:4.3e}')
            ax[_i].legend(_ls+[_l], ["Full-Order", "Truth"], loc=1)

        if ifpsp:
            # Pseudospectra
            zs = np.linspace(-4.0,0.5,51)
            ws = np.linspace(-6.6,6.6,51)
            gg = np.vstack([zs,ws])
            rng = np.array([0.25, 0.5])
            # Predicted
            pss, psk = [], []
            for _s in sas:
                grid, _pss = _s.estimate_ps(gg, mode='cont', method='standard', return_vec=False)
                grid, _psk = _s.estimate_ps(gg, mode='cont', method='sako', return_vec=False)
                pss.append((grid, _pss))
                psk.append((grid, _psk))

            for _i in range(Ns):
                grid, _pss = pss[_i]
                grid, _psk = psk[_i]
                fig, ax[_i] = complex_plot(grid, 1/_pss, rng, fig=(fig, ax[_i]), mode='line', lwid=2, lsty='dotted')
                fig, ax[_i] = complex_plot(grid, 1/_psk, rng, fig=(fig, ax[_i]), mode='line', lwid=1)

        for _i in range(2):
            ax[_i].set_xlim([-4.0, 0.5])
            ax[_i].set_ylim([-6.6, 6.6])

    if ifres:
        ## Residuals
        stys = ['bo', 'r^', 'gs', 'md', 'c*']
        fig, ax = plt.subplots()
        for _i in range(Ns):
            ax.semilogy(np.abs(sas[_i]._wd), sas[_i]._res, stys[_i], label=lbs[_i], markerfacecolor='none')
        ax.set_xlabel('Norm of eigenvalue')
        ax.set_ylabel('Residual')
        ax.legend()

    if ifspe:
        # Spectral measure
        stys = ['b-', 'r-', 'g--', 'm--', 'c-']
        def func_obs(x):
            _x1, _x2 = x.T
            return _x1+_x2
        vgs = []
        for _s in sas:
            _t, _v = _s.estimate_measure(func_obs, 6, 0.1, thetas=501)
            vgs.append((_t, _v))

        _arg = np.angle(wa)
        _amp = np.max(vgs[0][1])

        f = plt.figure()
        for _i in range(Ns):
            th, vg = vgs[_i]
            plt.plot(th, vg, stys[_i], label=lbs[_i], markerfacecolor='none')
        plt.plot([_arg[0], _arg[0]], [0, _amp], 'k:', label='System frequency')
        for _a in _arg[1:]:
            plt.plot([_a, _a], [0, _amp], 'k:')
        plt.legend()
        plt.xlabel('Angle, rad')
        plt.ylabel('Spectral measure')

    if ifegf:
        ## Eigenfunctions
        rngs = [[-2.5, 2.5], [-3.5, 3.5]]
        Ne = [101, 121]

        f1, a1 = plt.subplots(nrows=Ns, ncols=3, sharex=True, sharey=True, figsize=(7, 10))
        f2, a2 = plt.subplots(nrows=Ns, ncols=3, sharex=True, sharey=True, figsize=(7, 10))
        for i in range(Ns):
            _i1 = np.argmin(np.abs(sas[i]._wc - w0*1j))
            _i2 = np.argmin(np.abs(sas[i]._wc - 2*w0*1j))
            _i3 = np.argmin(np.abs(sas[i]._wc - 3*w0*1j))
            _idx = list(set([_i1, _i2, _i3]))
            sas[i].plot_eigfun_2d(rngs, Ne, _idx, mode='log', fig=(f1, a1[i]))
            a1[i][0].set_ylabel(lbs[i])
            sas[i].plot_eigfun_2d(rngs, Ne, _idx, mode='angle', fig=(f2, a2[i]))
            a2[i][0].set_ylabel(lbs[i])

            for _i in range(len(_idx)):
                a1[i][_i].plot(_ref[:,0], _ref[:,1], 'k--', linewidth=1)
                a2[i][_i].plot(_ref[:,0], _ref[:,1], 'k--', linewidth=1)

plt.show()
