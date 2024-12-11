from time import time
import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from keystone.src import KBF_ENC
from keystone.src import make_multi_step_loss, make_wf_smpl_loss, make_wf_jaco_loss, make_finite_diff_loss
from keystone.src import sample_unif_2d, plt_data
from keystone.src import Scaler, DataManager
from keystone.src import fit_model
from cases import solIVP_ci

POL_ORDER = 4  # Just testing for now

def run(CASE, MODE, INT_ORDER, NX, HORIZON_LENGTH, lr=0.001, NUM_EPOCHS=800, RESET_STOP=40, ext=''):
    # ----------------------------------
    # Parameters
    # ----------------------------------
    NUM_STEPS  = 20000
    NT = 25
    SHIFT = 1
    suf = f'{CASE}m{MODE}o{INT_ORDER}s{NX}h{HORIZON_LENGTH}'+ext

    ts = np.linspace(0, 2, NT+1)[:-1]
    ml = [-3.0, -2.0]

    # ----------------------------------
    # Model and Loss
    # ----------------------------------
    if CASE == 'CLO':
        make_loss = lambda m: make_wf_smpl_loss(m, HORIZON_LENGTH, INT_ORDER, ts[1])
    elif CASE == 'WLO':
        make_loss = lambda m: make_wf_jaco_loss(m, HORIZON_LENGTH, POL_ORDER, INT_ORDER, ts[1])
    elif CASE == 'FLO':
        assert HORIZON_LENGTH <= 5
        make_loss = lambda m: make_finite_diff_loss(m, INT_ORDER, ts[1])
    elif CASE == 'SLO':
        make_loss = make_multi_step_loss

    kbf = KBF_ENC([2, 3, 4], [16, 16], True, jax.nn.swish)

    # ----------------------------------
    # Data
    # ----------------------------------
    x0s = sample_unif_2d([[-5,5], [-5,5]], [NX, NX])
    tmp = np.array([
        solIVP_ci(ml, ts, _x0, None)
        for _x0 in x0s])
    scl = Scaler(Xs=tmp, mode='-11')
    sol = scl.vND(tmp)
    data = DataManager(sol, None, HORIZON_LENGTH, SHIFT, 16)
    scl.save(f'res/{suf}')

    # f = plt_data(data.data_train)
    # f.savefig(f'pics/dat.png', dpi=600)

    # ----------------------------------
    # Training
    # ----------------------------------
    lss = make_loss(kbf)
    optimizer = optax.adam(learning_rate=lr)

    if MODE == 'RST':
        RESET, STOP = True, RESET_STOP
    elif MODE == 'NON':
        RESET, STOP = False, 0
    elif MODE == 'INI':
        RESET, STOP = True, 0
    else:
        raise ValueError("Unknown mode {MODE}")
    fit = fit_model(
        f'res/{suf}', lss, data, NUM_EPOCHS, NUM_STEPS, 4,
        reset=RESET, reset_stop=STOP)
    ini = kbf.init_params()
    t1 = time()
    params, hist = fit(ini, optimizer)
    t2 = time()
    return t2-t1

if __name__ == "__main__":
    # ----------------------
    # Baseline comparison
    # ----------------------
    nx = 32

    # Single-step
    run('SLO', 'INI', 1, nx, 2, lr=0.001)
    # Multi-step
    run('SLO', 'INI', 1, nx, 5, lr=0.001)
    # # Multi-step
    # run('SLO', 'RST', 1, nx, 5, lr=0.001, NUM_EPOCHS=400)
    # CLO
    run('CLO', 'RST', 4, nx, 5, lr=0.001, NUM_EPOCHS=400)
    # # WLO
    # run('WLO', 'RST', 4, nx, 5, lr=0.001, NUM_EPOCHS=200)
    # # FLO
    # run('FLO', 'RST', 4, nx, 5, lr=0.001, NUM_EPOCHS=200)

    # Multi-step, varying horizon
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 21, 25]:
        run('SLO', 'RST', 1, nx, i, lr=0.001)

    # ----------------------
    # Parametric study of Weak Form
    # ----------------------
    case = 'CLO'
    order = 4
    nx = 32
    mode = 'RST'
    horizon = 25
    LR = 0.001

    # Mode
    for i in ['RST', 'NON', 'INI']:
        run(case, i, order, nx, horizon, lr=LR)

    # Effect of length of horizon, order=4
    for i in [5, 9, 13, 17, 21, 25]:
        run('CLO', mode, 4, nx, i, lr=LR)

    # ----------------------
    # Time cost benchmarking
    # ----------------------
    ls = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    t1 = []
    for i in ls:
        t = run('SLO', 'INI', 1, nx, i, lr=0.001, NUM_EPOCHS=5, ext='_ex')
        print(t)
        t1.append(t)
    t2 = []
    for i in ls:
        t = run(case, mode, 2, nx, i, lr=LR, NUM_EPOCHS=5, ext='_ex')
        print(t)
        t2.append(t)
    print(t1)
    print(t2)
    np.save(open(f'res/time.npy', 'wb'), [t1, t2])
