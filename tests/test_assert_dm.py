import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline
from scipy.special import erfinv

from dymad.numerics import DM, DMF, VBDM


def eig_plot(Eigenf, V):
    Q = np.linalg.solve(V.T @ V, V.T @ Eigenf)
    V_plot = V @ Q
    return V_plot


def compute_basis_and_errors(psi, theta, indices):
    _p = psi[:, indices]
    _k = indices[1] // 2
    _c = np.cos(_k * theta)
    _s = np.sin(_k * theta)
    c = np.linalg.lstsq(_p, np.hstack((_c, _s)))[0]
    basis = _p @ c
    error_cos = np.linalg.norm(basis[:, 0] - _c.reshape(-1))
    error_sin = np.linalg.norm(basis[:, 1] - _s.reshape(-1))
    return basis, (error_cos, error_sin)


def run_dm_s1(model, N=1000):
    # Parameters
    # 1 for fitting, 2 for interpolation
    theta1 = np.linspace(0, 2 * np.pi * (N - 1) / N, N).reshape(-1, 1)
    x1 = np.hstack((np.cos(theta1), np.sin(theta1)))
    theta2 = 2 * np.pi * np.sort(np.random.rand(400)).reshape(-1, 1)
    x2 = np.hstack((np.cos(theta2), np.sin(theta2)))

    k = int(np.sqrt(N))
    nvars = 10
    epsilon = None  # 1e-3
    alpha = 1

    if model == "dm":
        dm = DM(nvars, n_neighbors=k, alpha=alpha, epsilon=epsilon)
    elif model == "vbdm":
        dm = VBDM(n_components=nvars, n_neighbors=k, Kb=k // 2, operator="lb")
    elif model == "dmfk":
        dm = DMF(nvars, n_neighbors=k, alpha=alpha, epsilon=epsilon)
    else:
        dm = DMF(nvars, alpha=alpha, epsilon=epsilon)
    dm.fit(x1)
    eigval = dm._lambda
    eigvec1 = dm._psi
    eigvec2 = dm.transform(x2)

    psi1 = eigvec1 / np.sqrt(2)
    psi2 = eigvec2 / np.sqrt(2)

    return (psi1, theta1), (psi2, theta2), eigval


@pytest.mark.parametrize("model", ["dm", "dmf", "dmfk"])
def test_dm_s1(model):
    (psi1, theta1), (psi2, theta2), eigval = run_dm_s1(model, N=100)

    errors1, errors2 = [], []
    for i in range(4):
        _, error1 = compute_basis_and_errors(psi1, theta1, [2 * i + 1, 2 * (i + 1)])
        errors1.append(error1)
        _, error2 = compute_basis_and_errors(psi2, theta2, [2 * i + 1, 2 * (i + 1)])
        errors2.append(error2)
    errors1 = np.hstack(errors1)
    errors2 = np.hstack(errors2)

    if model == "dm":
        eps1, eps2, eps3 = 0.005, 0.01, 2e-3
    elif model == "dmfk":
        eps1, eps2, eps3 = 0.006, 0.02, 2e-3
    else:
        eps1, eps2, eps3 = 1e-13, 1e-13, 0.6

    assert np.max(errors1) < eps1, "DM S1 basis fitting"
    assert np.max(errors2) < eps2, "DM S1 basis interpolation"

    ref = np.array([0, 1, 1, 4, 4, 9, 9, 16, 16, 25])
    err = np.abs(eigval - ref) / np.maximum(1e-3, ref)
    assert np.max(err) < eps3, "DM S1 eigenvalues"


def run_vbdm(N=201, ifrand=1):
    K = int(np.sqrt(N)) * 10
    if ifrand:
        t = np.random.rand(N - 1) * 2 * np.pi
    else:
        t = np.linspace(0, 2 * np.pi, N)[:-1]
    X = np.vstack([np.cos(t), np.sin(t)]).T

    M = 201
    s = np.linspace(0, 2 * np.pi, M)
    Y = np.vstack([np.cos(s), np.sin(s)]).T

    vbdm = VBDM(n_components=6, n_neighbors=K, Kb=K // 2, operator="lb")
    vbdm.fit(X)
    ypsi, (yrho, yqes, ypeq) = vbdm.transform(Y, ret_den=True)
    ytmp = vbdm.transform(X, ret_den=False)

    IDX = np.argsort(t)
    rho_ref = UnivariateSpline(t[IDX], vbdm._rho[IDX], s=0)(s)
    qes_ref = UnivariateSpline(t[IDX], vbdm._qest[IDX], s=0)(s)
    peq_ref = UnivariateSpline(t[IDX], vbdm._peq[IDX], s=0)(s)

    return vbdm, (t, s), (ytmp, ypsi), (yrho, yqes, ypeq), (rho_ref, qes_ref, peq_ref)


def test_vbdm():
    vbdm, _, (ytmp, _), (yrho, yqes, ypeq), (rho_ref, qes_ref, peq_ref) = run_vbdm(N=401)

    err = np.linalg.norm(vbdm._psi - ytmp, axis=0) / np.linalg.norm(vbdm._psi, axis=0).mean()
    assert np.max(err[2:]) < 0.3, "VBDM eigenfunc on training data"

    assert np.linalg.norm(rho_ref - yrho) / np.linalg.norm(rho_ref) < 1e-2, "VBDM rho"
    assert np.linalg.norm(qes_ref - yqes) / np.linalg.norm(qes_ref) < 1e-2, "VBDM qes"
    assert np.linalg.norm(peq_ref - ypeq) / np.linalg.norm(peq_ref) < 1e-2, "VBDM peq"

    assert np.min(np.abs(vbdm._lambda + 1)) < 0.2, "VBDM lambda"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ifrand = 1

    ifdm = 1  # Includes VBDM too
    ifvb = 1
    ifs2 = 1  # Additional test for DM
    ifou = 1  # Additional test for VBDM

    if ifdm:
        models = ["dm", "dmf", "dmfk", "vbdm"]
        for _mdl in models:
            print(_mdl)
            (psi1, theta1), (psi2, theta2), eigval = run_dm_s1(_mdl, N=1000)

            ref = np.array([0, 1, 1, 4, 4, 9, 9, 16, 16, 25])
            if _mdl == "vbdm":
                eigval = -eigval

            print(eigval)
            print((eigval - ref) / np.maximum(1e-3, ref))

            # Compute and plot results
            _, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            errors1, errors2 = [], []
            titles = [r"$e^{ix}$", r"$e^{i2x}$", r"$e^{i3x}$", r"$e^{i4x}$"]

            for i, ax in enumerate(axes.flatten()):
                indices = [2 * i + 1, 2 * (i + 1)]
                basis1, error1 = compute_basis_and_errors(psi1, theta1, indices)
                errors1.append(error1)
                basis2, error2 = compute_basis_and_errors(psi2, theta2, indices)
                errors2.append(error2)

                ax.plot(theta1, np.cos((i + 1) * theta1), "r", label="Cosine")
                ax.plot(theta1, basis1[:, 0], "k--", label="Basis Cosine")
                ax.plot(theta2, basis2[:, 0], "g:", linewidth=2, label="Basis Cosine, Interp")
                ax.plot(theta1, np.sin((i + 1) * theta1), "r", label="Sine")
                ax.plot(theta1, basis1[:, 1], "k--", label="Basis Sine")
                ax.plot(theta2, basis2[:, 1], "g:", linewidth=2, label="Basis Sine, Interp")
                ax.set_xlim([0, 2 * np.pi])
                ax.set_ylim([-1, 1])
                ax.set_title(titles[i], fontsize=14)
            axes[0][0].legend()

            # Print errors
            print("Errors for theta1:", np.hstack(errors1))
            print("Errors for theta2:", np.hstack(errors2))

    if ifvb:
        vbdm, (t, s), (ytmp, ypsi), (yrho, yqes, ypeq), (rho_ref, qes_ref, peq_ref) = run_vbdm(
            N=201, ifrand=ifrand
        )

        err = np.linalg.norm(vbdm._psi - ytmp, axis=0) / np.linalg.norm(vbdm._psi, axis=0).mean()

        f, ax = plt.subplots(nrows=3, sharex=True)
        ax[0].plot(t, vbdm._rho, "b.")
        ax[0].plot(s, yrho, "r-")
        ax[1].plot(t, vbdm._qest, "b.")
        ax[1].plot(s, yqes, "r-")
        ax[2].plot(t, vbdm._peq, "b.")
        ax[2].plot(s, ypeq, "r-")

        I = np.argmin(t)
        f, ax = plt.subplots(nrows=6, sharex=True)
        for _i in range(6):
            _s = np.sign(vbdm._psi[I, _i])
            ax[_i].plot(t, vbdm._psi[:, _i] * _s, "b.", label="VBDM")
            ax[_i].plot(s, ypsi[:, _i] * _s, "r-", label="Interpolate")
            ax[_i].set_title(f"Error={err[_i]:4.3e}")
        ax[0].set_ylim([0, 2])
        ax[-1].set_xlabel(r"$\theta$")
        ax[-1].legend()

    if ifs2:
        # Generate the dataset
        N = 3000
        k = 246
        dim = 2
        alpha = 1
        nvars = 20
        epsilon = None  # 1e-3

        # Generate random spherical data
        test = np.random.rand(N, 2)
        THET = 2 * np.pi * test[:, 0]
        PHI = np.pi * test[:, 1]

        x = np.column_stack([np.sin(PHI) * np.cos(THET), np.sin(PHI) * np.sin(THET), np.cos(PHI)])

        # Step 1: Compute Diffusion Maps
        dm = DM(nvars, k, alpha=alpha, epsilon=epsilon)
        dm.fit(x)

        # True spherical data
        psitrue0 = x

        # Eigenfunction projection
        V_plot0 = np.zeros((psitrue0.shape[0], 3))
        V_plot0[:, :2] = eig_plot(psitrue0[:, :2], dm._psi[:, 1:3])
        V_plot0[:, 2] = eig_plot(
            psitrue0[:, 2].reshape(-1, 1), dm._psi[:, 3].reshape(-1, 1)
        ).reshape(-1)

        # Compute errors
        eigvectorerror0 = np.linalg.norm(V_plot0 - psitrue0, axis=0, ord=np.inf)

        # Plot eigenvectors
        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.scatter(THET, PHI, c=dm._psi[:, i + 1], s=5, cmap="viridis")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel(r"$\phi$", fontsize=14)
            plt.colorbar(label=f"Eigenvector {i + 1}")
            plt.xlim(0, 2 * np.pi)
            plt.ylim(0, np.pi)
        plt.tight_layout()

        # Generate testing data
        Ntheta2 = 75
        Nphi2 = 75
        theta2 = np.linspace(0, 2 * np.pi, Ntheta2, endpoint=False)
        phi2 = np.linspace(np.pi / (2 * Nphi2), np.pi - np.pi / (2 * Nphi2), Nphi2)
        THET2, PHI2 = np.meshgrid(theta2, phi2)
        THET2 = THET2.T.ravel()
        PHI2 = PHI2.T.ravel()

        x2 = np.column_stack(
            [np.sin(PHI2) * np.cos(THET2), np.sin(PHI2) * np.sin(THET2), np.cos(PHI2)]
        )

        # Visualize testing data
        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.scatter(THET2, PHI2, c=x2[:, i], s=5, cmap="viridis")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel(r"$\phi$", fontsize=14)
            plt.colorbar(label=f"Component {i + 1}")
            plt.xlim(0, 2 * np.pi)
            plt.ylim(0, np.pi)
        plt.tight_layout()

        # Step 2: Test Nyström on training data
        eigvecnyst = dm.transform(x, ifsym=True)

        eigvectorerror1 = np.linalg.norm(dm._psi[:, :3] - eigvecnyst[:, :3], axis=0, ord=np.inf)

        # Step 3: Test Nyström on testing data
        eigvecnyst = dm.transform(x2, ifsym=False)

        psitrue = x2
        V_plot = np.zeros((psitrue.shape[0], 3))
        V_plot[:, :2] = eig_plot(psitrue[:, :2], eigvecnyst[:, 1:3])
        V_plot[:, 2] = eig_plot(
            psitrue[:, 2].reshape(-1, 1), eigvecnyst[:, 3].reshape(-1, 1)
        ).reshape(-1)

        eigvectorerror2 = np.linalg.norm(V_plot - psitrue, axis=0, ord=np.inf)

        # Plot results of Nyström on testing data
        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.scatter(THET2, PHI2, c=V_plot[:, i], s=5, cmap="viridis")
            plt.xlabel(r"$\theta$", fontsize=14)
            plt.ylabel(r"$\phi$", fontsize=14)
            plt.colorbar(label=f"Projection {i + 1}")
            plt.xlim(0, 2 * np.pi)
            plt.ylim(0, np.pi)
        plt.tight_layout()

        # Print errors
        print("RMS error on training data (DM):", eigvectorerror0)
        print("RMS error on training data (Nyström):", eigvectorerror1)
        print("RMS error on testing data (Nyström):", eigvectorerror2)

        # Reference output:
        # RMS error on training data (DM): [0.37408906 1.02576904 0.99892442]
        # RMS error on training data (Nyström): [6.10622664e-15 7.99360578e-15 1.47659662e-14]
        # RMS error on testing data (Nyström): [0.45164767 1.018598   0.99906068]

    if ifou:
        N = 5000
        K0 = int(np.sqrt(N))
        if ifrand:
            t = np.random.randn(N)
            X = t.reshape(-1, 1)
        else:
            t = np.linspace(0, 1, N + 2)[1:-1]
            X = np.sqrt(2) * erfinv(2 * t - 1).reshape(-1, 1)

        # Eigenvalues
        ss = [2, 4, 6, 8, 10]
        ls = []
        for _s in ss:
            vbdm = VBDM(n_components=6, n_neighbors=K0 * _s, Kb=K0 * _s // 2, operator="kb")
            vbdm.fit(X)
            ls.append(vbdm._lambda)
        ls = np.vstack(ls).T

        lref = -np.arange(6)
        f = plt.figure()
        for _i in range(6):
            plt.plot(ss, ls[_i], "o-", markerfacecolor="none")
            plt.plot(ss, np.ones_like(ss) * lref[_i], "r--")

        f = plt.figure()
        plt.plot(X, vbdm._qest, "b.")
        plt.plot(X, vbdm._peq, "r.")

        # Interpolation
        M = 2001
        s = np.linspace(0, 1, M + 2)[1:-1]
        Y = np.sqrt(2) * erfinv(2 * s - 1)
        Y = np.sort(Y).reshape(-1, 1)

        ypsi, (yrho, yqes, ypeq) = vbdm.transform(Y, ret_den=True)

        f, ax = plt.subplots(nrows=3, sharex=True)
        ax[0].plot(X, vbdm._rho, "b.")
        ax[0].plot(Y, yrho, "r-")
        ax[1].plot(X, vbdm._qest, "b.")
        ax[1].plot(Y, yqes, "r-")
        ax[2].plot(X, vbdm._peq, "b.")
        ax[2].plot(Y, ypeq, "r-")

        # Eigenfunctions
        H = np.hstack(
            [
                np.ones_like(X),
                X,
                (X**2 - 1) / np.sqrt(2),
                (X**3 - 3 * X) / np.sqrt(6),
                (X**4 - 6 * X**2 + 3) / np.sqrt(24),
                (X**5 - 10 * X**3 + 15 * X) / np.sqrt(120),
            ]
        )

        I = 0
        f, ax = plt.subplots(nrows=6, sharex=True)
        for _i in range(6):
            _s = np.sign(H[I, _i])
            ax[_i].plot(X, H[:, _i], "go", markerfacecolor="none")
            ax[_i].plot(X, vbdm._psi[:, _i] * _s * np.sign(vbdm._psi[I, _i]), "b.")
            ax[_i].plot(Y, ypsi[:, _i] * _s * np.sign(ypsi[I, _i]), "r-")
        ax[0].set_ylim([0, 2])

    plt.show()
