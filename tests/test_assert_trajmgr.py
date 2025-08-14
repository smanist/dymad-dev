import copy
import numpy as np
from pathlib import Path

from dymad.data import DynData, TrajectoryManager
from dymad.utils import make_transform

HERE = Path(__file__).parent

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s.x, _t.x), f"{label} failed: {_s.x} != {_t.x}"
        assert np.allclose(_s.u, _t.u), f"{label} failed: {_s.u} != {_t.u}"
    print(f"{label} passed.")

def test_trajmgr(lti_data):
    metadata = {
        "config" : {
            "data": {
                "path": lti_data,
                "n_samples": 128,
                "n_steps": 501,
                "double_precision": False},
            "transform_x": [
                {
                "type": "Scaler",
                "mode" : "01"},
                {
                "type": "delay",
                "delay": 2}],
            "transform_u": {
                "type": "Scaler",
                "mode" : "-11"},
            "model": {
                "type": "NN"},
    }}

    # --------------------
    # First pass
    tm = TrajectoryManager(metadata, device="cpu")
    tm.process_all()

    data = np.load(lti_data, allow_pickle=True)
    xs = data['x']
    ts = data['t']
    us = data['u']

    trnx = make_transform(metadata['config']['transform_x'])
    trnu = make_transform(metadata['config']['transform_u'])

    xtrn = [xs[_i] for _i in tm.train_set_index]
    xtst = [xs[_i] for _i in tm.test_set_index]
    trnx.fit(xtrn)
    Xtst = trnx.transform(xtst)

    utrn = [us[_i] for _i in tm.train_set_index]
    utst = [us[_i] for _i in tm.test_set_index]
    trnu.fit(utrn)
    Utst = trnu.transform(utst)

    Dtst = [DynData(xt, ut[2:]) for xt, ut in zip(Xtst, Utst)]
    check_data(Dtst, tm.test_set, label='Transform X and U')

    Xrec = trnx.inverse_transform([_d.x for _d in Dtst])
    Urec = trnu.inverse_transform([_d.u for _d in Dtst])
    Drec = [DynData(xr, ur) for xr, ur in zip(Xrec, Urec)]
    Uref = [us[_i][2:] for _i in tm.test_set_index]
    Dref = [DynData(xt, ur) for xt, ur in zip(xtst, Uref)]
    check_data(Drec, Dref, label='Inverse Transform X and U')

    # --------------------
    # Second pass - reinitialize and reload
    old_metadata = copy.deepcopy(tm.metadata)

    new_tm = TrajectoryManager(old_metadata, device="cpu")
    new_tm.process_all()

    check_data(new_tm.train_set, tm.train_set, label="New Train")
    check_data(new_tm.valid_set, tm.valid_set, label="New Valid")
    check_data(new_tm.test_set,  tm.test_set,  label="New Test")
