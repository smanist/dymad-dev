import numpy as np

from dymad.transform.lift import poly_cross, poly_inverse, mixed_cross, mixed_inverse

def check_data(out, ref, label=''):
    for _s, _t in zip(out, ref):
        assert np.allclose(_s, _t), f"{label} failed: {_s} != {_t}"
    print(f"{label} passed.")

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

    # Tests
    Xt = poly_cross(Xs, Ks)
    check_data(Xt, Xp, label='Poly Cross')

    Xr = poly_inverse(Xt, Ks)
    check_data(Xr, Xs, label='Poly Inverse')

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

    # Tests
    opts = [
        (0, 'm', 3),
        (1, 'f', 2),
        (2, 'm', 4)
    ]

    Xt = mixed_cross(Xs, opts)
    check_data(Xt, Xp, label='Mixed Cross')

    Xr = mixed_inverse(Xt, opts)
    check_data(Xr, Xs, label='Mixed Inverse')

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

    # Tests
    opts = [
        (0, 'm', 5),
        (2, 'f', 2),
        ([3,1], 'p', [4,3])
    ]

    Xt = mixed_cross(Xs, opts)
    check_data(Xt, Xp, label='Mixed Cross')

    Xr = mixed_inverse(Xt, opts)
    check_data(Xr, Xs, label='Mixed Inverse')

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
