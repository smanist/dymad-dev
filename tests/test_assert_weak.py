import numpy as np

from dymad.utils.weak import generate_weak_weights

def test_weak_form():
    N  = 25
    x  = np.linspace(0.5, 2, N)
    dx = x[1]-x[0]
    y  = x*np.sin(x)
    dy = x*np.cos(x) + np.sin(x)

    for _o in range(1,5):
        C, D = generate_weak_weights(dx, N, _o, 4)

        diff = np.abs(C.dot(y) - D.dot(dy))

        assert np.all(diff[:2] < 1e-7), f"Weak weights failed for order {_o}: {diff.max()}"
        if len(diff) > 2:
            assert np.all(diff[2:] < 1e-5), f"Weak weights failed for order {_o}: {diff.max()}"
