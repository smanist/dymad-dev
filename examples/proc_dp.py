import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from keystone.src import KBF_ENC, Scaler, predict_ct
from keystone.src import plt_hist
from cases import solIVP_dp

FS = 12
FT = 6
FG = (6,3)
FQ = (3,3)

def load(INT_ORDER, NX, HORIZON_LENGTH):
    suf = f'DP_o{INT_ORDER}s{NX}h{HORIZON_LENGTH}'

    scl = Scaler(prev=f'res/{suf}')
    hist = pickle.load(open(f'res/{suf}_hst.pkl', 'rb'))
    params = pickle.load(open(f'res/{suf}_mdl.pkl', 'rb'))

    kbf = CKBF([4, 2, 9], [16, 32, 16], True, jax.nn.swish)

    return {
        'c' : suf,
        's' : scl,
        'h' : hist,
        'p' : params,
        'm' : kbf}

def gen_ref(ts, Nref):
    sols = []
    for _i in range(Nref):
        x0 = np.radians(20*(np.random.rand(4) - 0.5))
        sol = solIVP_dp(ts, x0, None)
        sols.append(sol)
    return sols

def pred(ts, sols, mdl):
    scl, kbf, params = mdl['s'], mdl['m'], mdl['p']
    Nref = len(sols)
    prd, err = [], []
    for _i in range(Nref):
        sol = sols[_i]
        snd = scl.ND(sol)
        x0 = snd[0,:4]
        ut = lambda t: sol[0,4:]
        xp = predict_dt(kbf, params, x0, ts, ut)
        tmp = np.hstack([xp, snd[:,4:]])
        xp = scl.DM(tmp)[:,:4]
        ee = np.linalg.norm(xp-sol[:,:4]) / np.linalg.norm(sol[:,:4])
        prd.append(xp)
        err.append(ee)
    return prd, err

def cmpTraj(ts, sols, stys, lbls):
    Nsol = len(sols)
    Nref = len(sols[0])
    f, ax = plt.subplots(nrows=4, sharex=True)
    for _i in range(Nsol):
        for _k in range(4):
            ax[_k].plot(ts, sols[_i][0][:,_k], stys[_i], label=lbls[_i])
    for _i in range(Nsol):
        for _j in range(1, Nref):
            for _k in range(4):
                ax[_k].plot(ts, sols[_i][_j][:,_k], stys[_i])
    plt.legend()
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
    ax[3].set_xlabel('Time', fontsize=FS)
    ax[0].set_ylabel(r'$\theta_1$', fontsize=FS)
    ax[1].set_ylabel(r'$\dot{\theta}_1$', fontsize=FS)
    ax[2].set_ylabel(r'$\theta_2$', fontsize=FS)
    ax[3].set_ylabel(r'$\dot{\theta}_2$', fontsize=FS)
    return f, acc

def batch_proc_hst(mdls, stys, lbls):
    f = plt.figure()
    for i in range(len(mdls)):
        plt_hist(mdls[i]['h'], ['L'], [stys[i]], lbls=[lbls[i]], avr=16, fig=f)
    return f

if __name__ == "__main__":
    STYS = ['b-', 'k--', 'g--', 'b:', 'k:', 'g:', 'b-.', 'k-.', 'g-.']
    Nt = 200
    ts = np.linspace(0, 4, Nt+1)[:-1]
    refs = gen_ref(ts, 100)

    # ----------------------
    # Parametric study of Weak Form
    # ----------------------
    order = 4
    nx = 320
    horizon = 25

    # Prediction
    lbls = ['BLO']
    mdls = [load(2, nx, 9)]
    f, err = batch_proc_prd(mdls, ts, refs[:5], STYS, lbls, ifplt=True)
    print(err)
    f.set_size_inches(*FG)
    plt.legend(fontsize=FS*0.8, bbox_to_anchor=(0.8, 0.7))
    f.savefig(f'pics/prd_dp.png', dpi=600, bbox_inches='tight')
