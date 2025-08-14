import numpy as np

from dymad.numerics.weak import generate_weak_weights

ref = [
    ([1.1692428919931415],
     [1.1656849312730149]),
    ([1.1665457945510198, -0.16746201240568834],
     [1.1665567588531893, -0.1674415417254616]),
    ([1.1665400003608295, -0.1674773324616382, -0.14860696787775846],
     [1.1665647114225761, -0.16743126605312575, -0.1485312656603982]),
    ([1.166550437535383, -0.16744975524314526, -0.14866118895602212, -0.0003130890072928494],
     [1.1665503785832487, -0.16744976558358238, -0.14865772262988253, -0.0003051354281532214])
]

def test_weak_form():
    N  = 25
    x  = np.linspace(0.5, 2, N)
    dx = x[1]-x[0]
    y  = x*np.sin(x)
    dy = x*np.cos(x) + np.sin(x)

    for _o in range(1,5):
        C, D = generate_weak_weights(dx, N, _o, _o)

        assert np.allclose(C.dot(y), ref[_o-1][0]), f"Weak weights failed for order {_o}: C*y"
        assert np.allclose(D.dot(dy), ref[_o-1][1]), f"Weak weights failed for order {_o}: D*dy"
