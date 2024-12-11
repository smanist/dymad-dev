import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

from keystone.src import KBF_ENC, Scaler, predict_ct, predict_dt
from keystone.src import plt_hist
from cases import solIVP_ci

FS = 12
FT = 6
FG = (6,3)
FQ = (3,3)

def load(CASE, MODE, INT_ORDER, NX, HORIZON_LENGTH):
    suf = f'{CASE}m{MODE}o{INT_ORDER}s{NX}h{HORIZON_LENGTH}'

    scl = Scaler(prev=f'res/{suf}')
    hist = pickle.load(open(f'res/{suf}_hst.pkl', 'rb'))
    params = pickle.load(open(f'res/{suf}_mdl.pkl', 'rb'))

    kbf = KBF_ENC([2, 3, 4], [16, 16], True, jax.nn.swish)

    return {
        'c' : suf,
        's' : scl,
        'h' : hist,
        'p' : params,
        'm' : kbf}

def gen_ref(ts, Nref):
    ml = [-3.0, -2.0]
    sols = []
    for _i in range(Nref):
        x0 = 10*(np.random.rand(2) - 0.5)
        sol = solIVP_ci(ml, ts, x0, None)
        sols.append(sol)
    return sols

def pred(ts, sols, mdl):
    suf, scl, kbf, params = mdl['c'], mdl['s'], mdl['m'], mdl['p']
    Nref = len(sols)
    prd, err = [], []
    for _i in range(Nref):
        sol = sols[_i]
        snd = scl.ND(sol)
        x0 = snd[0,:2]
        if suf[:3] == 'SLO':
            ur = snd[:,2:]
            xp = predict_dt(kbf, params, x0, ur)
        else:
            ut = lambda t: snd[0,2:]
            xp = predict_ct(kbf, params, x0, ts, ut)
        tmp = np.hstack([xp, snd[:,2:]])
        xp = scl.DM(tmp)[:,:2]
        ee = np.linalg.norm(xp-sol[:,:2]) / np.linalg.norm(sol[:,:2])
        prd.append(xp)
        err.append(ee)
    return prd, err

def cmpTraj(ts, sols, stys, lbls):
    Nsol = len(sols)
    Nref = len(sols[0])
    f, ax = plt.subplots(nrows=2, sharex=True)
    for _i in range(Nsol):
        for _k in range(2):
            ax[_k].plot(ts, sols[_i][0][:,_k], stys[_i], label=lbls[_i])
    for _i in range(Nsol):
        for _j in range(1, Nref):
            for _k in range(2):
                ax[_k].plot(ts, sols[_i][_j][:,_k], stys[_i])
    return f

def batch_proc_prd(mdls, ts, refs, stys, lbls, ifplt=True):
    prds, errs = [], []
    for i in range(len(mdls)):
        prd, err = pred(ts, refs, mdls[i])
        prds.append(prd)
        errs.append(err)
    acc = np.array([np.mean(_e) for _e in errs])
    if not ifplt:
        return acc
    f = cmpTraj(
        ts, prds+[refs], stys[:len(lbls)]+['r--'], lbls+['Truth'])
    plt.legend(fontsize=FS)
    ax = f.get_axes()
    ax[1].set_xlabel('Time', fontsize=FS)
    ax[0].set_ylabel('$x_1$', fontsize=FS)
    ax[1].set_ylabel('$x_2$', fontsize=FS)
    return f, acc

def batch_proc_hst(mdls, stys, lbls, ifnrm=False):
    f = plt.figure()
    for i in range(len(mdls)):
        plt_hist(mdls[i]['h'], ['L'], [stys[i]], lbls=[lbls[i]], avr=16, fig=f, ifnrm=ifnrm)
    plt.legend(fontsize=FS)
    xt = np.array([0, 200, 400, 600, 800])
    ax = plt.gca()
    ax.set_xlabel('Epochs', fontsize=FS)
    ax.set_xticks(xt*16)
    ax.set_xticklabels([str(_) for _ in xt])
    ax.tick_params(labelsize=FS)
    return f

if __name__ == "__main__":
    STYS = ['b-', 'k--', 'g--', 'b:', 'k:', 'g:', 'b-.', 'k-.', 'g-.']
    Nt = 25
    ts = np.linspace(0, 2, Nt+1)[:-1]
    refs = gen_ref(ts, 100)
    Nr = 5

    ifbas = 0
    ifhrz = 1
    iftim = 0
    ifmod = 0

    # ----------------------
    # Baseline comparison
    # ----------------------
    nx = 32
    if ifbas:
        rclo = load('CLO', 'RST', 4, nx, 5)  # CLO
        # rsng = load('SLO', 'INI', 1, nx, 2)  # Single-step
        rsng = load('SLO', 'RST', 1, nx, 2)  # Single-step
        rmul = load('SLO', 'RST', 1, nx, 5)  # Multi-step
        mdls = [rclo, rsng, rmul]
        lbls = ['BLO', 'SLO: 1-step', 'SLO: 5-step']
        f = batch_proc_hst(mdls, STYS, lbls, ifnrm=True)
        plt.ylabel('Normalized Loss', fontsize=FS)
        f.set_size_inches(*FQ)
        f.savefig(f'pics/hst_basel.png', dpi=600, bbox_inches='tight')
        err = batch_proc_prd(mdls, ts, refs, STYS, lbls, ifplt=False)
        print(err)
        f, _ = batch_proc_prd(mdls, ts, refs[:Nr], STYS, lbls, ifplt=True)
        plt.legend(fontsize=FS*0.8, bbox_to_anchor=(0.5, 0.5))
        f.set_size_inches(*FQ)
        f.savefig(f'pics/prd_basel.png', dpi=600, bbox_inches='tight')

    if ifhrz:
        # Effect of length of horizon, for SLO
        l1 = [2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 21, 25]
        lbls = [str(l) for l in l1]
        mdls = [load('SLO', 'RST', 1, nx, l) for l in l1]
        stys = ['b-', 'b--', 'b-.', 'r-', 'r--', 'r-.', 'k-', 'k--', 'k-.', 'g-', 'g--', 'g-.']
        f = batch_proc_hst(mdls, stys, lbls, ifnrm=True)
        plt.ylabel('Normalized Loss', fontsize=FS)
        f.set_size_inches(*FQ)
        plt.ylim([2e-7,2])
        plt.legend(fontsize=FS*0.8)
        f.savefig(f'pics/hst_horiz_slo.png', dpi=600, bbox_inches='tight')
        err1 = batch_proc_prd(mdls, ts, refs, STYS, lbls, ifplt=False)
        print(err1)

        # Effect of length of horizon, for BLO, order=4
        l2 = [5, 9, 13, 17, 21, 25]
        lbls = [str(l) for l in l2]
        mdls = [load('CLO', 'RST', 4, 32, l) for l in l2]
        stys = ['b-', 'b--', 'b-.', 'r-', 'r--', 'r-.', 'k-']
        f = batch_proc_hst(mdls, stys, lbls, ifnrm=True)
        plt.ylabel('Normalized Loss', fontsize=FS)
        f.set_size_inches(*FQ)
        plt.ylim([2e-7,2])
        plt.legend(fontsize=FS*0.8)
        f.savefig(f'pics/hst_horiz_o4.png', dpi=600, bbox_inches='tight')
        err2 = batch_proc_prd(mdls, ts, refs, STYS, lbls, ifplt=False)
        print(err2)

        f = plt.figure(figsize=FQ)
        plt.semilogy(l1, err1*100, 'bo-', markerfacecolor='none', label='SLO')
        plt.semilogy(l2, err2*100, 'rs-', markerfacecolor='none', label='BLO')
        plt.legend(fontsize=FS)
        plt.xlabel('Horizon length', fontsize=FS)
        plt.ylabel('Prediction error, %', fontsize=FS)
        ax = plt.gca()
        ax.set_xticks(l2)
        ax.set_xticklabels([str(_) for _ in l2])
        ax.tick_params(labelsize=FS)
        f.savefig(f'pics/err_horiz.png', dpi=600, bbox_inches='tight')

    if iftim:
        ls = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])
        t1, t2 = np.load(open(f'res/time.npy', 'rb'))[:,:-3]
        tr = t1/t2
        rf = (tr[-2]-tr[0])/(ls[-2]-ls[0])*(ls-ls[0])+tr[0]

        f = plt.figure(figsize=FQ)
        plt.plot(ls, tr, 'k^-', label='Relative cost', markerfacecolor='none')
        plt.plot(ls, rf, 'r--', label='First-order reference')
        plt.xlabel('Horizon length', fontsize=FS)
        plt.ylabel('Cost(SLO)/Cost(BLO)', fontsize=FS)
        plt.legend(fontsize=FS*0.8)
        ax = plt.gca()
        ax.tick_params(labelsize=FS)
        f.savefig(f'pics/time_horiz.png', dpi=600, bbox_inches='tight')

    # ----------------------
    # Parametric study of BLO
    # ----------------------
    case = 'CLO'
    order = 4
    nx = 32
    mode = 'RST'
    horizon = 25

    if ifmod:
        # Mode
        cass = ['NON', 'INI', 'RST']
        lbls = ['None', 'Initial', 'BLO']
        mdls = [load(case, m, order, nx, horizon) for m in cass]
        f = batch_proc_hst(mdls, STYS, lbls)
        plt.ylabel('Loss', fontsize=FS)
        f.set_size_inches(*FG)
        f.savefig(f'pics/hst_start.png', dpi=600, bbox_inches='tight')
        # f, err = batch_proc_prd(mdls[:2], ts, refs, STYS, lbls[:2])
        # print(err)
        # f.savefig(f'pics/prd_start.png', dpi=600)
