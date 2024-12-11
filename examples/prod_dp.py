import jax.numpy as jn
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from keystone.src import KBF_ENC
from keystone.src import make_wf_smpl_loss
from keystone.src import sample_unif_2d, plt_data
from keystone.src import Scaler, DataManager
from keystone.src import fit_model
from cases import solIVP_dp

def run(INT_ORDER, NX, HORIZON_LENGTH, lr=0.001):
    # ----------------------------------
    # Parameters
    # ----------------------------------
    NUM_EPOCHS = 800
    NUM_STEPS  = 20000
    NT = 25
    SHIFT = 1
    suf = f'DP_o{INT_ORDER}s{NX}h{HORIZON_LENGTH}'

    ts = np.linspace(0, 2, NT+1)[:-1]

    # ----------------------------------
    # Model and Loss
    # ----------------------------------
    make_loss = lambda m: make_wf_smpl_loss(m, HORIZON_LENGTH, INT_ORDER, ts[1])
    kbf = KBF_ENC([4, 2, 9], [32, 32, 32], True, jax.nn.swish)

    # ----------------------------------
    # Data
    # ----------------------------------
    x0s = np.radians(20*(np.random.rand(NX,4) - 0.5))
    tmp = np.array([
        solIVP_dp(ml, ts, _x0, None)
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

    fit = fit_model(f'res/{suf}', lss, data, NUM_EPOCHS, NUM_STEPS, 4, reset=True, init_only=False)
    ini = kbf.init_params()
    params, hist = fit(ini, optimizer)

if __name__ == "__main__":
    order = 4
    nx = 320
    horizon = 9
    LR = 0.0001

    run(order, nx, horizon, lr=LR)
