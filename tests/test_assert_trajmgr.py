import copy
import numpy as np
from pathlib import Path
import pytest
import torch

from dymad.io import TrajectoryManager
from dymad.io.legacy_runtime import LegacyRuntimeBatch
from dymad.transform import make_transform

HERE = Path(__file__).parent

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s.t, _t.t), f"{label} failed: {_s.t} != {_t.t}"
        assert np.allclose(_s.x, _t.x), f"{label} failed: {_s.x} != {_t.x}"
        assert np.allclose(_s.y, _t.y), f"{label} failed: {_s.y} != {_t.y}"
        assert np.allclose(_s.u, _t.u), f"{label} failed: {_s.u} != {_t.u}"
        assert np.allclose(_s.p, _t.p), f"{label} failed: {_s.p} != {_t.p}"
    print(f"{label} passed.")

def test_dyndata(trj_data):
    data = np.load(trj_data, allow_pickle=True)
    ts = torch.tensor(data['t'])
    xs = torch.tensor(data['x'])
    ys = torch.tensor(np.tile(data['y'], (8, 1, 1)))
    us = torch.tensor(np.tile(data['u'], (1, 21, 1)))
    ps = torch.tensor(data['p'])

    Dlist = [
        LegacyRuntimeBatch(t=t, x=x, y=y, u=u, p=p)
        for t, x, y, u, p in zip(ts, xs, ys, us, ps)
    ]
    assert Dlist[0].n_steps == ts.size(1), "n_steps in single traj"
    assert Dlist[0].batch_size == 1, "batch_size in single traj"

    data = LegacyRuntimeBatch.collate(Dlist)
    dref = LegacyRuntimeBatch(t=ts, x=xs, y=ys, u=us, p=ps)
    check_data([data], [dref], label='LegacyRuntimeBatch collate')
    assert data.n_steps == ts.size(1), "n_steps in collated traj"
    assert data.batch_size == ts.size(0), "batch_size in collated traj"
    assert dref.n_steps == ts.size(1), "n_steps in batched traj"
    assert dref.batch_size == ts.size(0), "batch_size in batched traj"

    N = 5
    trun = data.truncate(N)
    dref = LegacyRuntimeBatch(t=ts[:,:N], x=xs[:,:N], y=ys[:,:N], u=us[:,:N], p=ps)
    check_data([trun], [dref], label='LegacyRuntimeBatch truncate')

    trun = data.get_step(N,N+2)
    dref = LegacyRuntimeBatch(t=ts[:,N:N+2], x=xs[:,N:N+2], y=ys[:,N:N+2], u=us[:,N:N+2], p=ps)
    check_data([trun], [dref], label='LegacyRuntimeBatch get_step two steps')

    trun = data.get_step(N)
    dref = LegacyRuntimeBatch(t=ts[:,N:N+1], x=xs[:,N], y=ys[:,N], u=us[:,N], p=ps)
    check_data([trun], [dref], label='LegacyRuntimeBatch get_step one step')

    W, S = 5, 10
    ufld = data.unfold(W, S)
    dref = LegacyRuntimeBatch(
        t = ts.unfold(1, W, S).reshape(-1, W),
        x = xs.unfold(1, W, S).reshape(-1, xs.size(-1), W).permute(0, 2, 1),
        y = ys.unfold(1, W, S).reshape(-1, ys.size(-1), W).permute(0, 2, 1),
        u = us.unfold(1, W, S).reshape(-1, us.size(-1), W).permute(0, 2, 1),
        p = ps.repeat_interleave(((xs.size(1)-W)//S +1), dim=0)
    )
    check_data([ufld], [dref], label='LegacyRuntimeBatch unfold')

@pytest.mark.parametrize("case", [0, 1])
def test_trajmgr(trj_data, case):
    metadata = {
        "config" : {
            "data": {
                "path": trj_data,
                "n_samples": 8,
                "n_steps": 21,
                "double_precision": False},
            "data_valid": {
                "path": trj_data,
                "n_samples": 4,
                "n_steps": 15,
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
            # transform_y will be Identity
            "transform_p": {
                "type": "Scaler",
                "mode" : "std"},
    }}

    # --------------------
    # First pass
    # --------------------
    if case == 0:
        # Using (as if) two datasets
        # Allow valid to be different length
        t_valid = 15
        metadata['config']['data_valid']['n_steps'] = t_valid
        # train set is loaded, preprocessed, and transformed
        trn = TrajectoryManager(metadata, data_key="train", device="cpu")
        trn.process_all()
        # valid set is loaded, preprocessed, but not transformed
        vld = TrajectoryManager(metadata, data_key="valid", device="cpu")
        vld.prepare_data()
        vld.set_data_index()
        # valid set gets transforms from train set
        vld.set_transforms(trajmgr=trn)
        vld.process_data()
    elif case == 1:
        # Using just one dataset, but split into train/valid
        trn_idx = [0, 2, 4, 5, 6]
        vld_idx = [1, 3, 7]
        # This time both train and valid must have same length
        t_valid = 21
        metadata['config']['data_valid']['n_steps'] = t_valid
        # train set is loaded, preprocessed, and transformed
        trn = TrajectoryManager(metadata, data_key="train", device="cpu")
        trn.prepare_data()
        trn.set_data_index(trn_idx)
        trn.process_data()
        # train set is loaded, preprocessed, but not transformed
        # Note that here we instantiate another TrajectoryManager
        vld = TrajectoryManager(metadata, data_key="train", device="cpu")
        vld.prepare_data()
        vld.set_data_index(vld_idx)
        # valid set gets transforms from train set
        vld.set_transforms(trajmgr=trn)
        vld.process_data()
    else:
        raise ValueError(f"Unknown case {case}")

    # Make up reference data
    data = np.load(trj_data, allow_pickle=True)
    ts = data['t']
    xs = data['x']
    ys = data['y']
    us = np.tile(data['u'], (1, 21, 1))
    ps = data['p']

    trnx = make_transform(metadata['config']['transform_x'])
    trnu = make_transform(metadata['config']['transform_u'])
    trnp = make_transform(metadata['config']['transform_p'])

    ttst = [ts[_i][:t_valid] for _i in vld.data_index]

    xtrn = [xs[_i] for _i in trn.data_index]
    xtst = [xs[_i][:t_valid] for _i in vld.data_index]
    trnx.fit(xtrn)
    Xtst = trnx.transform(xtst)

    Ytst = [ys[:t_valid] for _ in vld.data_index]

    utrn = [us[_i] for _i in trn.data_index]
    utst = [us[_i][:t_valid] for _i in vld.data_index]
    trnu.fit(utrn)
    Utst = trnu.transform(utst)

    ptrn = [ps[_i] for _i in trn.data_index]
    ptst = [ps[_i] for _i in vld.data_index]
    trnp.fit(ptrn)
    Ptst = trnp.transform(ptst)

    Dtst = [
        LegacyRuntimeBatch(t=tt[2:], x=xt, y=yt[2:], u=ut[2:], p=pt)
        for tt, xt, yt, ut, pt in zip(ttst, Xtst, Ytst, Utst, Ptst)
        ]
    check_data(Dtst, vld.dataset, label='Transform')

    Xrec = trnx.inverse_transform([_d.x[0] for _d in Dtst])
    Yrec = [_d.y[0] for _d in Dtst]
    Urec = trnu.inverse_transform([_d.u[0] for _d in Dtst])
    Prec = trnp.inverse_transform([_d.p for _d in Dtst])
    Drec = [
        LegacyRuntimeBatch(t=tt, x=xr, y=yr, u=ur, p=pr)
        for tt, xr, yr, ur, pr in zip(ttst, Xrec, Yrec, Urec, Prec)
    ]

    yref = [ys[2:t_valid] for _ in vld.data_index]
    uref = [us[_i][2:t_valid] for _i in vld.data_index]
    Dref = [
        LegacyRuntimeBatch(t=tr, x=xr, y=yr, u=ur, p=pr)
        for tr, xr, yr, ur, pr in zip(ttst, xtst, yref, uref, ptst)
    ]
    check_data(Drec, Dref, label='Inverse Transform')

    # --------------------
    # Second pass - reinitialize and reload
    # --------------------
    trn_metadata = copy.deepcopy(trn.metadata)
    vld_metadata = copy.deepcopy(vld.metadata)

    new_trn = TrajectoryManager(trn_metadata, device="cpu")
    new_vld = TrajectoryManager(vld_metadata, device="cpu")
    new_trn.process_all()
    new_vld.process_all()

    check_data(new_trn.dataset, trn.dataset, label="New Train")
    check_data(new_vld.dataset, vld.dataset, label="New Valid")
