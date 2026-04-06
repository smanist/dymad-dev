import numpy as np

from dymad.numerics import check_direction, complex_grid, complex_plot, make_random_matrix
from dymad.sako import RALowRank, estimate_pseudospectrum, resolvent_analysis

Ndim = 20
Nrnk = 6


def make_mat(dt):
    Ar, Er = make_random_matrix(Ndim, Nrnk, [-2, 0], [1, 2], dt=dt)
    er, Vr = np.linalg.eig(Ar)
    idx = np.argsort(np.abs(er))[::-1]
    er = er[idx]
    Vr = Vr[:, idx]
    return Ar, Er, er, Vr


def plot_conv(dt, Ar, Er, er, Vr):
    Nitr = Nrnk + 1
    s_ref = Er[0][0] * 1.1

    ## Exact
    gre, ure, vre = resolvent_analysis(s_ref, A=Ar, B=None, ord=Nrnk, return_vec=True)

    ## Constrained
    gr = np.zeros((Nitr, Nrnk))
    ur = np.zeros((Nitr, Nrnk))
    vr = np.zeros((Nitr, Nrnk))
    for _i in range(Nitr):
        grc, urc, vrc = resolvent_analysis(
            s_ref, A=Ar, B=None, U=Vr[:, : _i + 2], ord=Nrnk, return_vec=True
        )
        gr[_i, : _i + 2] = grc
        for _j in range(min(_i + 2, Nrnk)):
            ur[_i, _j] = check_direction(ure[:, _j], urc[:, _j])
            vr[_i, _j] = check_direction(vre[:, _j], vrc[:, _j])

    rng = np.arange(0, Nitr) + 2
    plt.figure()
    for _i in range(Nrnk):
        plt.plot(rng, gr[:, _i])
        plt.plot([rng[0], rng[-1]], [gre[_i], gre[_i]], "b--")
    plt.xlabel("Subspace dimension")
    plt.ylabel("Resolvent gains")

    plt.figure()
    for _i in range(Nrnk):
        plt.plot(rng, ur[:, _i], "--")
        plt.plot(rng, vr[:, _i], "-")
    plt.xlabel("Subspace dimension")
    plt.ylabel("Vector alignment")


def plot_pspc(dt, Ar, Er, er, Vr):
    if dt < 0:
        zs = np.linspace(-2.5, 0.5, 41)
        ws = np.linspace(-3, 3, 41)
        grid = complex_grid(np.vstack([zs, ws]))
        rng = np.array([0.1, 0.25])
    else:
        Rs = np.linspace(-0.1, 1.1, 41)
        Is = np.linspace(-0.7, 0.7, 41)
        grid = complex_grid(np.vstack([Rs, Is]))
        rng = np.array([0.1, 0.25]) * dt
    ge = estimate_pseudospectrum(grid, resolvent_analysis, return_vec=False, A=Ar, B=None, ord=1)
    g0 = estimate_pseudospectrum(
        grid, resolvent_analysis, return_vec=False, A=Ar, B=None, U=Vr[:, :2], ord=1
    )
    g1 = estimate_pseudospectrum(
        grid, resolvent_analysis, return_vec=False, A=Ar, B=None, U=Vr[:, :4], ord=1
    )
    g2 = estimate_pseudospectrum(
        grid, resolvent_analysis, return_vec=False, A=Ar, B=None, U=Vr[:, :Nrnk], ord=1
    )

    f, ax = plt.subplots()
    ax.plot(er.real, er.imag, "bo", markerfacecolor="none")
    f, ax = complex_plot(grid, 1 / ge, rng, fig=(f, ax), mode="line", lwid=1)
    f, ax = complex_plot(grid, 1 / g0, rng, fig=(f, ax), mode="line", lsty="dotted")
    f, ax = complex_plot(grid, 1 / g1, rng, fig=(f, ax), mode="line", lsty="dashdot")
    f, ax = complex_plot(grid, 1 / g2, rng, fig=(f, ax), mode="line", lsty="dashed")


def run_ras(Ns):
    dt = 0.02

    _, _Er = make_random_matrix(Ndim, Nrnk, [-2, 0], [1, 2], dt=-1)
    L0, U0, V0 = _Er
    Ad = U0.dot(np.diag(np.exp(L0 * dt))).dot(V0.conj().T)
    _tmp = (np.exp(L0 * dt) - 1) / L0
    Bd = U0.dot(np.diag(_tmp)).dot(V0.conj().T)

    Ac = U0.dot(np.diag(L0)).dot(V0.conj().T)
    Bc = U0.dot(V0.conj().T)

    rals = RALowRank(U0, np.diag(L0), V0, dt=dt)

    zs = np.linspace(-2.5, 0.5, Ns)
    ws = np.linspace(-3, 3, Ns)
    grid = complex_grid(np.vstack([zs, ws]))
    grdt = np.exp(grid * dt)

    gd, ud, vd = estimate_pseudospectrum(
        grdt, resolvent_analysis, return_vec=True, A=Ad, B=Bd, ord=1
    )
    gc, uc, vc = estimate_pseudospectrum(
        grid, resolvent_analysis, return_vec=True, A=Ac, B=Bc, ord=1
    )  # Differ by zero eigenvalues
    ge, ue, ve = estimate_pseudospectrum(
        grid, resolvent_analysis, return_vec=True, A=Ac, U=U0, ord=1
    )  # Same as RALS
    gr, ur, vr = estimate_pseudospectrum(grid, rals, return_vec=True)

    ras = [(gd, ud, vd), (gc, uc, vc), (ge, ue, ve), (gr, ur, vr)]
    return grid, L0, ras


def test_ras():
    grid, _, ras = run_ras(6)
    gc, uc, vc = ras[1]
    ge, ue, ve = ras[2]
    gr, ur, vr = ras[3]

    msk = np.abs(grid) > 0.2

    assert 1 - np.min(check_direction(ue[:, msk], uc[:, msk])) < 1e-14
    assert 1 - np.min(check_direction(ve[:, msk], vc[:, msk])) < 1e-14
    assert 1 - np.min(check_direction(ue, ur)) < 1e-14
    assert 1 - np.min(check_direction(ve, vr)) < 1e-14

    assert np.linalg.norm(ge[msk] - gc[msk]) < 1e-11
    assert np.linalg.norm(ge - gr) < 1e-11


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Continuous time
    # Should see at least top 3-4 gains and vec align converge at subspace dim = Nrnk
    # In pseudospectra, g2 and ge should overlap, g1 close, g0 less close
    dt = -1
    Ar, Er, er, Vr = make_mat(dt)
    plot_conv(dt, Ar, Er, er, Vr)
    plot_pspc(dt, Ar, Er, er, Vr)

    # Discrete time
    # Should see similar convergence, but most of them (4+)
    # Similar in pseudospectra
    dt = 0.2
    Ar, Er, er, Vr = make_mat(dt)
    plot_conv(dt, Ar, Er, er, Vr)
    plot_pspc(dt, Ar, Er, er, Vr)

    # RALS
    # Most contours should overlap
    # The ones from discrete-time can be slightly off or quite off
    grid, L0, ras = run_ras(61)
    gd, ud, vd = ras[0]
    gc, uc, vc = ras[1]
    ge, ue, ve = ras[2]
    gr, ur, vr = ras[3]

    f, ax = plt.subplots()
    ax.plot(L0.real, L0.imag, "bo", markerfacecolor="none")
    f, ax = complex_plot(grid, 1 / gd, None, fig=(f, ax), mode="line", lwid=1)
    f, ax = complex_plot(grid, 1 / gc, None, fig=(f, ax), mode="line", lsty="dotted")
    f, ax = complex_plot(grid, 1 / ge, None, fig=(f, ax), mode="line", lsty="dashed")
    f, ax = complex_plot(grid, 1 / gr, None, fig=(f, ax), mode="line", lwid=1, lsty="dotted")

    plt.show()
