import pickle

import matplotlib.pyplot as plt
import numpy as np

from dymad.utils import adj_to_edge

# ------------------------
# Parameters for users
# ------------------------
# Parameters for Kuramoto model
wl, wu = 1.0, 15.0  # Lower and upper bounds of intrinsic frequency

## Lower and upper bounds of coupling strength, choose one case below
kl, ku = 4.0, 6.0  # de = 0.75
# kl, ku = 2., 3.            # de = 0.50
# kl, ku = 1., 1.5           # de = 0.19

# Time settings
# Ntrj   = 200               # Number of trajectories
# nStint = 10                # Number of time intervals
# Nt     = 100
Ntrj = 20  # Number of trajectories
nStint = 20  # Number of time intervals
Nt = 100

# Model size
nNodes = 2
nSubSys = 3
nStates = nNodes * nSubSys

# Control flow
iftop = 0  # Generate topologies
ifdyn = 1  # Generate dynamics data
ifplt = 1  # Plot dynamics
# ------------------------
# End of Parameters for users
# ------------------------

# File name suffix
suf = f"_n{nNodes}_s{nSubSys}_k{int(kl)}_s{nStint}"

if iftop:
    # Topology generation
    def genAdj(nNodes):
        _A = np.random.rand(nNodes, nNodes)
        _A = (_A.T + _A) / 2
        _A = _A * (ku - kl) + kl
        _A -= np.diag(np.diag(_A))
        return _A

    ## Coefficients
    W = np.random.rand(nStates) * (wu - wl) + wl  # Diagonal, intrinsic frequency
    A = np.array([genAdj(nNodes) for _ in range(nSubSys)])  # Intra-group interaction
    K = np.random.rand(nSubSys - 1) * (ku - kl) + kl  # Inter-group interaction

    ## Intra-group part
    AA = np.zeros((nStates, nStates))
    for n in range(nSubSys):
        idx, idy = n * nNodes, (n + 1) * nNodes
        AA[idx:idy, idx:idy] = A[n]
    AA -= np.diag(np.diag(AA))
    AA += np.diag(W)

    ## Inter-group part
    ### Using a random chain structure as baseline
    idx = np.arange(0, nSubSys)
    np.random.shuffle(idx)
    ### Location of inter-group connections
    ldx = np.random.randint(0, nNodes, size=(nSubSys - 1, 2))
    ### Masks for random topology - One full chain, and then one edge removed per case
    msk = [np.ones(nSubSys - 1, dtype=bool) for _ in range(nSubSys)]
    for i in range(nSubSys - 1):
        msk[i + 1][i] = False

    ### Now assemble the inter-group connections, generating nSubSys different topologies
    AAs = []
    for n in range(nSubSys):
        A0 = AA.copy()
        for m in range(nSubSys - 1):
            if msk[n][m]:
                i, j = ldx[m]
                I = idx[m] * nNodes
                J = idx[m + 1] * nNodes
                A0[I + i, J + j] = K[m]
                A0[J + j, I + i] = K[m]
        AAs.append(A0)

    ## Save figures of topologies for sanity check
    vmn1, vmx1 = np.min(AAs[0]), np.max(AAs[0])
    f, ax = plt.subplots(1, nSubSys, figsize=(4 * nSubSys, 4))
    for n in range(nSubSys):
        im0 = ax[n].imshow(AAs[n], vmin=vmn1, vmax=vmx1)
        ax[n].set_title(f"Topology {n + 1}")
        ax[n].set_xticks(np.arange(0, nStates + 1, 1) - 0.5)
        ax[n].set_yticks(np.arange(0, nStates + 1, 1) - 0.5)
        ax[n].grid(which="both", color="white", linewidth=0.5)
        ax[n].set_xticklabels([])
        ax[n].set_yticklabels([])
    cbar = f.colorbar(im0, ax=ax.tolist(), shrink=0.9, orientation="vertical")
    ax[0].set_ylabel("Full")
    plt.savefig(f"./data/topology{suf}.png")

    np.savez(f"./data/topology{suf}.npz", AAs=AAs)

if ifdyn:
    topo = np.load(f"./data/topology{suf}.npz", allow_pickle=True)
    AAs = topo["AAs"]
    EIs, EWs = adj_to_edge(AAs)
    Aidx = np.random.randint(0, nSubSys, size=(Ntrj, nStint))

    SEED = np.arange(0, 20 * Ntrj, 20)
    xs, e1, e2 = [], [], []
    for j in range(Ntrj):
        np.random.seed(SEED[j])

        # Initialize randomly and solve for multiple intervals
        # The IC is discarded later
        X0 = np.random.rand(nStates).reshape(1, -1) * 2 - 1
        trj, eis, ews = [X0], [], []
        for i in range(nStint):
            for _k in range(Nt // nStint):
                tmp = trj[-1].dot(AAs[Aidx[j][i]]) / (wu * 1.15)
                trj.append(tmp)
                eis.append(EIs[Aidx[j][i]])
                ews.append(EWs[Aidx[j][i]])
        xs.append(np.vstack(trj[:-1]))  # (nStint*nTime, nStates)
        e1.append(eis)  # (nStint*nTime, nEdges, ...)
        e2.append(ews)  # (nStint*nTime, nEdges, ...)

    with open(f"./data/data{suf}.pkl", "wb") as f:
        pickle.dump({"x": xs, "ei": e1, "ew": e2, "Aidx": Aidx}, f)

if ifplt:
    data = np.load(f"./data/data{suf}.pkl", allow_pickle=True)
    xs = data["x"]
    Aidx = data["Aidx"]

    Ntrj = len(xs)
    Nt, Nx = xs[0].shape
    T = np.arange(Nt)

    f, ax = plt.subplots(Nx + 1, 1, sharex=True, figsize=(8, Nx + 1))
    ax = ax.flatten()
    idx = Aidx[0]
    tdx = np.arange(0, len(idx) - 1) + 1
    tdx = tdx * (Nt // nStint)
    for n in range(Nx):
        for t in tdx:
            ax[n].axvline(t, color="k", linestyle="--")
    for i in range(Ntrj):
        _AIdx = np.hstack([np.ones(Nt // nStint) * _i for _i in Aidx[i]])
        for n in range(Nx):
            ax[n].plot(xs[i][:, n], label=f"SubSys {n + 1}")
            ax[n].set_ylabel(f"State {n + 1}")
        ax[-1].plot(T, _AIdx + 1)
        ax[-1].set_ylabel("Topology")

    plt.tight_layout()
    # plt.savefig(f'./data/trajectory{suf}.png')
    plt.show()
