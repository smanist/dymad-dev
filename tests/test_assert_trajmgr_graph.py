import copy
import numpy as np
import pytest
import torch

from dymad.io import TrajectoryManagerGraph
from dymad.io.data import DynData
from dymad.transform import make_transform
from dymad.utils import adj_to_edge

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s.t, _t.t), f"{label} failed: {_s.t} != {_t.t}"
        assert np.allclose(_s.x, _t.x), f"{label} failed: {_s.x} != {_t.x}"
        assert _s.y is None
        assert _t.y is None
        assert np.allclose(_s.u, _t.u), f"{label} failed: {_s.u} != {_t.u}"
        assert np.allclose(_s.p, _t.p), f"{label} failed: {_s.p} != {_t.p}"
        for _s_ei, _t_ei in zip(_s.ei, _t.ei):
            assert np.allclose(_s_ei, _t_ei), f"{label} failed: {_s_ei} != {_t_ei}"
        for _s_ew, _t_ew in zip(_s.ew, _t.ew):
            assert np.allclose(_s_ew, _t_ew), f"{label} failed: {_s_ew} != {_t_ew}"
        assert _s.ea is None
        assert _t.ea is None
    print(f"{label} passed.")

def test_dyndata_graph(ltg_data):
    data = np.load(ltg_data, allow_pickle=True)
    ts = torch.tensor(data['t'][:2, :4])
    xs = torch.tensor(data['x'][:2, :4])
    us = torch.tensor(data['u'][:2, :4])
    ps = torch.tensor(data['p'][:2])

    e1 = torch.tensor([[0,1,2],[1,0,0]]).transpose(0,1)
    w1 = torch.tensor([1.0, 1.0, 1.0])
    e2 = torch.tensor([[0,2],[1,0]])
    w2 = torch.tensor([1.0, 3.0])
    es = [[e1, e2, e2, e1],
          [e2, e1, e2, e1]]
    ws = [[w1, w2, w2, w1],
          [w2, w1, w2, w1]]

    Dlist = [
        DynData(t=t, x=x, u=u, p=p, ei=e, ew=w)
        for t, x, u, p, e, w in zip(ts, xs, us, ps, es, ws)
    ]
    assert Dlist[0].n_steps == ts.size(1), "n_steps in single traj"
    assert Dlist[0].batch_size == 1, "batch_size in single traj"

    data = DynData.collate(Dlist)
    dref = DynData(t=ts, x=xs, u=us, p=ps, ei=es, ew=ws)
    check_data([data], [dref], label='DynData graph collate')
    assert data.n_steps == ts.size(1), "n_steps in collated traj"
    assert data.batch_size == ts.size(0), "batch_size in collated traj"
    assert dref.n_steps == ts.size(1), "n_steps in batched traj"
    assert dref.batch_size == ts.size(0), "batch_size in batched traj"

    N = 2
    trun = data.truncate(N)
    dref = DynData(
        t=ts[:,:N], x=xs[:,:N], u=us[:,:N], p=ps,
        ei=[_e[:N] for _e in es], ew=[_w[:N] for _w in ws])
    check_data([trun], [dref], label='DynData graph truncate')

    trun = data.get_step(N,N+2)
    dref = DynData(
        t=ts[:,N:N+2], x=xs[:,N:N+2], u=us[:,N:N+2], p=ps,
        ei=[_e[N:N+2] for _e in es], ew=[_w[N:N+2] for _w in ws])
    check_data([trun], [dref], label='DynData graph get_step two steps')

    trun = data.get_step(N)
    dref = DynData(
        t=ts[:,N:N+1], x=xs[:,N:N+1], u=us[:,N:N+1], p=ps,
        ei=[[_e[N]] for _e in es], ew=[[_w[N]] for _w in ws])
    check_data([trun], [dref], label='DynData graph get_step one step')

    W, S = 3, 1
    ufld = data.unfold(W, S)
    dref = DynData(
        t = torch.vstack([ts[:,:3], ts[:,1:]]),
        x = torch.vstack([xs[:,:3], xs[:,1:]]),
        u = torch.vstack([us[:,:3], us[:,1:]]),
        p = torch.vstack([ps, ps]),
        ei = [es[0][:3], es[1][:3], es[0][1:], es[1][1:]],
        ew = [ws[0][:3], ws[1][:3], ws[0][1:], ws[1][1:]]
    )

    check_data([ufld], [dref], label='DynData graph unfold')

@pytest.mark.parametrize("case", [0, 1])
def test_trajmgr_graph(ltg_data, case):
    DLY = 2
    metadata = {
        "config" : {
            "data": {
                "path": ltg_data,
                "n_samples": 32,
                "n_steps": 51,
                "double_precision": False},
            "data_valid": {
                "path": ltg_data,
                "n_samples": 32,
                "n_steps": 15,
                "double_precision": False},
            "transform_x": [
                {
                "type": "Scaler",
                "mode" : "01"},
                {
                "type": "delay",
                "delay": DLY}],
            "transform_u": {
                "type": "Scaler",
                "mode" : "-11"},
            "transform_p": {
                "type": "Scaler",
                "mode" : "std"},
            "transform_ew": {
                "type": "Scaler",
                "mode" : "-11"},
    }}

    adj = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [1, 2, 0]
    ])
    edge_index, edge_weights = adj_to_edge(adj)

    # --------------------
    # First pass
    # --------------------
    if case == 0:
        # Using (as if) two datasets
        # Allow valid to be different length
        t_valid = 15
        metadata['config']['data_valid']['n_steps'] = t_valid
        # train set is loaded, preprocessed, and transformed
        trn = TrajectoryManagerGraph(metadata, data_key="train", adj=adj, device="cpu")
        trn.process_all()
        # valid set is loaded, preprocessed, but not transformed
        vld = TrajectoryManagerGraph(metadata, data_key="valid", adj=adj, device="cpu")
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
        t_valid = 51
        metadata['config']['data_valid']['n_steps'] = t_valid
        # train set is loaded, preprocessed, and transformed
        trn = TrajectoryManagerGraph(metadata, data_key="train", adj=adj, device="cpu")
        trn.prepare_data()
        trn.set_data_index(trn_idx)
        trn.process_data()
        # train set is loaded, preprocessed, but not transformed
        # Note that here we instantiate another TrajectoryManager
        vld = TrajectoryManagerGraph(metadata, data_key="train", adj=adj, device="cpu")
        vld.prepare_data()
        vld.set_data_index(vld_idx)
        # valid set gets transforms from train set
        vld.set_transforms(trajmgr=trn)
        vld.process_data()
    else:
        raise ValueError(f"Unknown case {case}")

    # Make up reference data
    data = np.load(ltg_data, allow_pickle=True)
    ts = data['t']
    xs = data['x']
    us = data['u']
    ps = data['p']

    # ---
    # Forward transform test
    # ---
    # Manually transform data
    # Transform only one node, and then replicate it for all nodes
    trnx = make_transform(metadata['config']['transform_x'])
    trnu = make_transform(metadata['config']['transform_u'])
    trnp = make_transform(metadata['config']['transform_p'])
    trne = make_transform(metadata['config']['transform_ew'])

    ttst = [ts[_i][:t_valid] for _i in vld.data_index]

    xtrn = [xs[_i][:,:2] for _i in trn.data_index]
    xtst = [xs[_i][:t_valid,:2] for _i in vld.data_index]
    trnx.fit(xtrn)
    Xtst = trnx.transform(xtst)
    Xtst = [np.concatenate([xt, xt, xt], axis=-1) for xt in Xtst]

    utrn = [us[_i][:,0].reshape(-1,1) for _i in trn.data_index]
    utst = [us[_i][:t_valid,0].reshape(-1,1) for _i in vld.data_index]
    trnu.fit(utrn)
    Utst = trnu.transform(utst)
    Utst = [np.concatenate([ut, ut, ut], axis=-1) for ut in Utst]

    ptrn = [ps[_i][0] for _i in trn.data_index]
    ptst = [ps[_i][0] for _i in vld.data_index]
    trnp.fit(ptrn)
    Ptst = trnp.transform(ptst)
    Ptst = [np.concatenate([pt, pt, pt], axis=-1) for pt in Ptst]

    itst = [[edge_index for _ in range(len(xs[_i]))] for _i in vld.data_index]

    etrn = [[edge_weights for _ in range(len(xs[_i]))] for _i in trn.data_index]
    etst = [[edge_weights for _ in range(len(xs[_i]))] for _i in vld.data_index]
    trne.fit([np.hstack(e).reshape(-1,1) for e in etrn])
    Etst = []
    for e in etst:
        ew = trne.transform([_e.reshape(-1,1) for _e in e])
        Etst.append([e.reshape(-1) for e in ew])

    Dtst = [
        DynData(
            t=torch.tensor(tt[DLY:]),
            x=torch.tensor(xt),
            u=torch.tensor(ut[DLY:]),
            p=torch.tensor(pt),
            ei=[torch.tensor(e) for e in ei[DLY:t_valid]],
            ew=[torch.tensor(e) for e in ew[DLY:t_valid]])
        for tt, xt, ut, pt, ei, ew in zip(ttst, Xtst, Utst, Ptst, itst, Etst)
        ]
    check_data(Dtst, vld.dataset, label='Graph Transform')

    # ---
    # Inverse transform test
    # ---
    # Again manually inverse transform data
    # Inverse transform only one node, and then replicate it for all nodes
    Xrec = trnx.inverse_transform([_d.x[0,:,:2*(DLY+1)] for _d in Dtst])
    Xrec = [np.concatenate([xr, xr, xr], axis=-1) for xr in Xrec]
    Urec = trnu.inverse_transform([_d.u[0,:,0].reshape(-1,1) for _d in Dtst])
    Urec = [np.concatenate([ur, ur, ur], axis=-1) for ur in Urec]
    Prec = trnp.inverse_transform([_d.p[0,:4] for _d in Dtst])
    Erec = []
    for _d in Dtst:
        ew = trne.inverse_transform([_e.reshape(-1,1) for _e in _d.ew])
        Erec.append([e.reshape(-1) for e in ew])
    Drec = [
        DynData(
            t=torch.tensor(tt),
            x=torch.tensor(xr),
            u=torch.tensor(ur),
            p=pr.clone().detach(),
            ei=[torch.tensor(e) for e in ei],
            ew=[e.clone().detach() for e in ew])
        for tt, xr, ur, pr, ei, ew in zip(ttst, Xrec, Urec, Prec, itst, Erec)
    ]

    Xref = [xs[_i][:t_valid] for _i in vld.data_index]
    Uref = [us[_i][DLY:t_valid] for _i in vld.data_index]
    Pref = [ps[_i] for _i in vld.data_index]
    Eref = [[edge_weights for _ in range(len(xs[_i][:t_valid]))] for _i in vld.data_index]
    Dref = [
        DynData(
            t=torch.tensor(tr),
            x=torch.tensor(xr),
            u=torch.tensor(ur),
            p=torch.tensor(pr),
            ei=[torch.tensor(e) for e in ei[DLY:t_valid]],
            ew=[torch.tensor(e) for e in ew[DLY:t_valid]])
        for tr, xr, ur, pr, ei, ew in zip(ttst, Xref, Uref, Pref, itst, Eref)
        ]
    check_data(Drec, Dref, label='Graph Inverse Transform')

    # --------------------
    # Second pass - reinitialize and reload
    trn_metadata = copy.deepcopy(trn.metadata)
    vld_metadata = copy.deepcopy(vld.metadata)

    new_trn = TrajectoryManagerGraph(trn_metadata, adj=adj, device="cpu")
    new_vld = TrajectoryManagerGraph(vld_metadata, adj=adj, device="cpu")
    new_trn.process_all()
    new_vld.process_all()

    check_data(new_trn.dataset, trn.dataset, label="New Train")
    check_data(new_vld.dataset, vld.dataset, label="New Valid")
