import numpy as np

from dymad.transform.base import Lift
from dymad.transform.lift import poly_cross, poly_inverse, mixed_cross, mixed_inverse

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"
    print(f"{label} passed.")

def check_class(inp, out, label, fobs, finv, **kwargs):
    # First pass
    lift = Lift(fobs, finv, **kwargs)
    lift.fit(inp)
    Xt = lift.transform(inp)
    if out is not None:
        check_data(Xt, out, label=f'{label} Transform (Lift)')
    Xr = lift.inverse_transform(Xt)
    check_data(Xr, inp, label=f'{label} Inverse (Lift)')

    # Reload test
    stt = lift.state_dict()
    reld = Lift()
    reld.load_state_dict(stt)
    Xt = reld.transform(inp)
    if out is not None:
        check_data(Xt, out, label=f'{label} Transform (Lift/Reload)')
    Xr = reld.inverse_transform(Xt)
    check_data(Xr, inp, label=f'{label} Inverse (Lift/Reload)')

def test_poly():
    # Test data
    Xs = np.array([
            [1.,  2., -0.1],
            [1.1, 3., -0.2],
            [1.2, 4., -0.3],
            [1.3, 5., -0.4]])
    Ks = [3, 2, 4]

    # Manual computation
    Xp = []
    _x1, _x2, _x3 = Xs.T
    for k1 in range(Ks[0]):
        for k2 in range(Ks[1]):
            for k3 in range(Ks[2]):
                Xp.append((_x1**k1)*(_x2**k2)*(_x3**k3))
    Xp = np.vstack(Xp).T

    # Tests - Raw
    Xt = poly_cross(Xs, Ks)
    check_data(Xt, Xp, label='Poly Cross')

    Xr = poly_inverse(Xt, Ks)
    check_data(Xr, Xs, label='Poly Inverse')

    # Tests - Lift class
    check_class([Xs, Xs], [Xp, Xp], 'Poly', 'poly', None, Ks=Ks)

def test_mixed_mf():
    # Test data
    Xs = np.array([
            [1.,  0.4, -0.1],
            [1.1, 0.3, -0.2],
            [1.2, 0.2, -0.3],
            [1.3, 0.1, -0.4]])
    Ks = [3, 2, 4]

    # Manual computation
    Xp = []
    _x1, _x2, _x3 = Xs.T
    _p = [
        np.ones_like(_x2),
        np.cos(_x2),
        np.sin(_x2),
        np.cos(2*_x2),
        np.sin(2*_x2)]
    for k1 in range(Ks[0]):
        for k2 in range(2*Ks[1]+1):
            for k3 in range(Ks[2]):
                Xp.append((_x1**k1)*_p[k2]*(_x3**k3))
    Xp = np.vstack(Xp).T

    # Tests - Raw
    opts = [
        (0, 'm', 3),
        (1, 'f', 2),
        (2, 'm', 4)
    ]

    Xt = mixed_cross(Xs, opts)
    check_data(Xt, Xp, label='Mixed Cross')

    Xr = mixed_inverse(Xt, opts)
    check_data(Xr, Xs, label='Mixed Inverse')

    # Tests - Lift class
    check_class([Xs, Xs], [Xp, Xp], 'Mixed', 'mixed', None, opts=opts)

def test_mixed_mfp():
    # Test data
    Xs = np.array([
            [1.,   0.4, -0.1, 2.],
            [1.1,  0.3, -0.2, 2.1],
            [1.2, -0.2, -0.3, 2.2],
            [1.3, -0.1, -0.4, 2.3]])
    Ks = [5, 3, 2, 4]

    # Manual computation
    Xp = []
    _x1, _x2, _x3, _x4 = Xs.T
    _r = np.sqrt(_x2**2 + _x4**2)
    _t = np.arctan2(_x2, _x4)
    _p1 = [
        np.ones_like(_x3),
        np.cos(_x3),
        np.sin(_x3),
        np.cos(2*_x3),
        np.sin(2*_x3)]
    _p2 = [
        np.ones_like(_t),
        np.cos(_t),
        np.sin(_t),
        np.cos(2*_t),
        np.sin(2*_t),
        np.cos(3*_t),
        np.sin(3*_t)]
    for k1 in range(Ks[0]):
        for k2 in range(2*Ks[1]+1):
            for k3 in range(2*Ks[2]+1):
                for k4 in range(Ks[3]):
                    Xp.append((_x1**k1)*_p2[k2]*_p1[k3]*(_r**k4))
    Xp = np.vstack(Xp).T

    # Tests - Raw
    opts = [
        (0, 'm', 5),
        (2, 'f', 2),
        ([3,1], 'p', [4,3])
    ]

    Xt = mixed_cross(Xs, opts)
    check_data(Xt, Xp, label='Mixed Cross')

    Xr = mixed_inverse(Xt, opts)
    check_data(Xr, Xs, label='Mixed Inverse')

    # Tests - Lift class
    check_class([Xs, Xs], [Xp, Xp], 'Mixed', 'mixed', None, opts=opts)

def test_mixed_more():
    Xs = np.random.rand(10, 6)
    opts = [
        (0, 'm', 5),
        ([2,4], 'p', [2,3]),
        ([3,1], 'p', [4,3]),
        (5, 'f', 3)
    ]

    Xt = mixed_cross(Xs, opts)
    Xr = mixed_inverse(Xt, opts)
    check_data(Xr, Xs, label='Mixed More Inverse')

    check_class([Xs, Xs], None, 'Mixed', 'mixed', None, opts=opts)

def test_custom_finv():
    def fobs(x, a=1.0):
        return np.vstack([x[:,0], np.exp(a*x[:,1])]).T

    def finv(z, a=1.0):
        return np.vstack([z[:,0], np.log(z[:,1])/a]).T

    Xs = np.array([
            [1.,  0.4],
            [1.1, 0.3],
            [1.2, 0.2],
            [1.3, 0.1]])
    Xp = np.array([
            [1.,  np.exp(0.4)],
            [1.1, np.exp(0.3)],
            [1.2, np.exp(0.2)],
            [1.3, np.exp(0.1)]])

    check_class([Xs, Xs], [Xp, Xp], 'Custom finv', fobs, finv, a=1.0)

def test_custom_pinv():
    def fobs(x):
        return np.vstack([x[:,0], x[:,0]+x[:,1]]).T

    Xs = np.array([
            [1.,  0.4],
            [1.1, 0.3],
            [1.2, 0.2],
            [1.3, 0.1]])
    Xp = np.array([
            [1.,  1.4],
            [1.1, 1.4],
            [1.2, 1.4],
            [1.3, 1.4]])

    check_class([Xs, Xs], [Xp, Xp], 'Custom pinv', fobs, None)
