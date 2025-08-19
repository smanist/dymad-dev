import numpy as np

from dymad.numerics.spectrum import generate_coef, rational_kernel

def cmp(sol, ref):
    return np.linalg.norm(sol-ref) / np.linalg.norm(ref)

c3ref = np.array([
    (-202+79*1j)/80,
    121/20,
    (-202-79*1j)/80,
])  # eps=0.1
d3ref = np.array([-2-1j, 5, -2+1j])

c4ref = np.array([
    (-1165710-2944643*1j)/750000,
    (513570+3570527*1j)/250000,
    (513570-3570527*1j)/250000,
    (-1165710+2944643*1j)/750000,
])  # eps=0.1
d4ref = np.array([
    (-39+65*1j)/24,
    (17-85*1j)/8,
    (17+85*1j)/8,
    (-39-65*1j)/24,
])

def test_generate_coef():
    t3, z3, c3, d3 = generate_coef(3, 0.1)
    t4, z4, c4, d4 = generate_coef(4, 0.1)

    eps = 2.4e-15
    assert cmp(c3, c3ref) < eps
    assert cmp(d3, d3ref) < eps
    assert cmp(c4, c4ref) < eps
    assert cmp(d4, d4ref) < eps

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ths = np.linspace(-np.pi, np.pi, 201)
    f = plt.figure()
    for _i in range(1, 7):
        plt.plot(ths, rational_kernel(ths, _i, 1.0).real, label=f'm={_i}')
    plt.legend()
    plt.xlabel(r'$\theta$')
    plt.title(r'$Re(K_1(\theta))$')

    plt.show()