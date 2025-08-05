import numpy as np

from dymad.data import Compose, DelayEmbedder, Identity, make_transform, Scaler

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"

Xs = [
    np.array([
        [1., 2.],
        [1.1, 3.],
        [1.2, 4.],
        [1.3, 5.],
        [1.4, 6.],
        [1.5, 7.]]),
    np.array([
        [2.2, 3.4],
        [2.3, 3.5],
        [2.4, 3.6],
        [2.5, 3.7]]),
    np.array([
        [1.,  2.5],
        [1.2, 4.5],
        [1.4, 6.5],
        [1.6, 8.5]])]
Xn = np.array([
    [1.32, 2.4],
    [1.33, 3.5],
    [1.34, 4.6],
    [1.35, 5.7]])

# Identity
iden = Identity()
Xt = iden.transform(Xs)
check_data(Xt, Xs, label='Identity')

Xi = iden.inverse_transform(Xt)
check_data(Xi, Xs, label='Inverse Identity')

# Scaling
sclr = Scaler(mode='01')
sclr.fit(Xs)
Xt = sclr.transform([Xn])[0]

tmp = np.vstack(Xs)
mx, mn = np.max(tmp, axis=0), np.min(tmp, axis=0)
Xr = (Xn - mn) / (mx - mn)
check_data(Xt, Xr, label='Scalar 01')

Xi = sclr.inverse_transform([Xt])[0]
check_data(Xi, Xn, label='Inverse Scalar 01')

# Delay embedding
dely = DelayEmbedder(delay=2)
dely.fit(Xs)
Xt = dely.transform([Xn])[0]

Xr = np.vstack([
    Xn[:3].reshape(1, -1),
    Xn[1:4].reshape(1, -1)])
check_data(Xt, Xr, label='Delay')

Xi = dely.inverse_transform([Xt])[0]
check_data(Xi, Xn, label='Inverse Delay')

# Compose
cmps = Compose([Scaler(mode='std'), DelayEmbedder(delay=1)])
cmps.fit(Xs)
Xt = cmps.transform([Xn])[0]

tmp = np.vstack(Xs)
avr, std = np.mean(tmp, axis=0), np.std(tmp, axis=0)
tmp = (Xn - avr) / std
Xr = np.vstack([
    tmp[:2].reshape(1, -1),
    tmp[1:3].reshape(1, -1),
    tmp[2:4].reshape(1, -1)])
check_data(Xt, Xr, label='Compose')

Xi = cmps.inverse_transform([Xt])[0]
check_data(Xi, Xn, label='Inverse Compose')

# Interface
mktr = make_transform([
    {'type': 'scaler', 'mode': 'std'},
    {'type': 'delay', 'delay': 1}
])
mktr.fit(Xs)
Xt = mktr.transform([Xn])[0]
check_data(Xt, Xr, label='Compose make_transform')

Xi = mktr.inverse_transform([Xt])[0]
check_data(Xi, Xn, label='Inverse Compose make_transform')

# Loading and reloading
dct = mktr.state_dict()

reld = Compose()
reld.load_state_dict(dct)

Xt = reld.transform([Xn])[0]
check_data(Xt, Xr, label='Compose reload')

Xi = reld.inverse_transform([Xt])[0]
check_data(Xi, Xn, label='Inverse Compose reload')
