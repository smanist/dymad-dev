import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.integrate import solve_ivp

#------------------------
# Parameters for users
#------------------------
# Parameters for Kuramoto model
wl, wu = 1., 15.           # Lower and upper bounds of intrinsic frequency

## Lower and upper bounds of coupling strength, choose one case below
kl, ku = 4., 6.            # de = 0.75
# kl, ku = 2., 3.            # de = 0.50
# kl, ku = 1., 1.5           # de = 0.19

# Time settings
Ntrj   = 10                # Number of trajectories
split  = 0.8               # Train/test split ratio
nStint = 5                 # Number of time intervals
dt     = 0.01              # Time step size
tf     = 10.               # Final time

# Model size
nNodes  = 4
nSubSys = 5
nStates = nNodes * nSubSys

# Control flow
iftop = 0                  # Generate topologies
ifdyn = 1                  # Generate dynamics data
ifplt = 1                  # Plot dynamics
#------------------------
# End of Parameters for users
#------------------------

# File name suffix
suf = f'_n{nNodes}_s{nSubSys}_k{int(kl)}_s{nStint}'

if iftop:
    # Topology generation
    def genAdj(nNodes):
        _A = np.random.rand(nNodes,nNodes)
        _A = (_A.T + _A) / 2
        _A = _A * (ku-kl) + kl
        _A -= np.diag(np.diag(_A))
        return _A

    ## Coefficients
    W = np.random.rand(nStates) * (wu-wl) + wl          # Diagonal, intrinsic frequency
    A = np.array([genAdj(nNodes) for _ in range(nSubSys)])   # Intra-group interaction
    K = np.random.rand(nSubSys-1) * (ku-kl) + kl        # Inter-group interaction

    ## Intra-group part
    AA = np.zeros((nStates,nStates))
    for n in range(nSubSys):
        idx,idy = n*nNodes,(n+1)*nNodes
        AA[idx:idy,idx:idy] = A[n]
    AA -= np.diag(np.diag(AA))

    ## Inter-group part
    ### Using a random chain structure as baseline
    idx = np.arange(0,nSubSys)
    np.random.shuffle(idx)
    ### Location of inter-group connections
    ldx = np.random.randint(0,nNodes,size=(nSubSys-1,2))
    ### Masks for random topology - One full chain, and then one edge removed per case
    msk = [np.ones(nSubSys-1, dtype=bool) for _ in range(nSubSys)]
    for i in range(nSubSys-1):
        msk[i+1][i] = False

    ### Now assemble the inter-group connections, generating nSubSys different topologies
    AAs = []
    for n in range(nSubSys):
        A0 = AA.copy()
        for m in range(nSubSys-1):
            if msk[n][m]:
                i, j = ldx[m]
                I = idx[m]*nNodes
                J = idx[m+1]*nNodes
                A0[I+i, J+j] = K[m]
                A0[J+j, I+i] = K[m]
        AAs.append(A0)

    ### Also compute the coarsened version
    CCs = [_A.reshape(nSubSys, nNodes, nSubSys, nNodes).mean(axis=(1,3)) for _A in AAs]

    ## Save figures of topologies for sanity check
    ref = AAs[0] + np.diag(W)
    vmn1, vmx1 = np.min(ref), np.max(ref)
    vmn2, vmx2 = np.min(CCs[0]), np.max(CCs[0])
    f, ax = plt.subplots(2, nSubSys, figsize=(4*nSubSys,8))
    for n in range(nSubSys):
        tmp = AAs[n] + np.diag(W)
        im0 = ax[0,n].imshow(tmp, vmin=vmn1, vmax=vmx1)
        im1 = ax[1,n].imshow(CCs[n], vmin=vmn2, vmax=vmx2)
        ax[0,n].set_title(f'Topology {n+1}')
        ax[0,n].set_xticks(np.arange(0, nStates+1, 1)-0.5)
        ax[0,n].set_yticks(np.arange(0, nStates+1, 1)-0.5)
        ax[1,n].set_xticks(np.arange(0, nSubSys+1, 1)-0.5)
        ax[1,n].set_yticks(np.arange(0, nSubSys+1, 1)-0.5)
        for _i in [0, 1]:
            ax[_i,n].grid(which='both', color='white', linewidth=0.5)
            ax[_i,n].set_xticklabels([])
            ax[_i,n].set_yticklabels([])
    cbar = f.colorbar(im0, ax=ax[0].tolist(), shrink=0.9, orientation='vertical')
    cbar = f.colorbar(im1, ax=ax[1].tolist(), shrink=0.9, orientation='vertical')
    ax[0,0].set_ylabel('Full')
    ax[1,0].set_ylabel('Coarsened')
    plt.savefig(f'./data/topology{suf}.png')

    np.savez(f'./data/topology{suf}.npz', W=W, AAs=AAs, CCs=CCs)

if ifdyn:
    topo = np.load(f'./data/topology{suf}.npz', allow_pickle=True)
    W = topo['W']
    AAs = topo['AAs']
    CCs = topo['CCs']
    Aidx = np.random.randint(0, nSubSys, size=(Ntrj, nStint))

    t0, t1 = 0, tf/nStint
    T = np.arange(t0, t1, dt)  # Time for one interval
    def kuramotoODE(t, x, W, A):
        _diff_x = x-x[:,None]
        return W + np.sum(A * np.sin(_diff_x),axis=1)

    SEED = np.arange(0,20*Ntrj,20)
    xs, ys, us, adj = [], [], [], []
    for j in range(Ntrj):
        np.random.seed(SEED[j])

        # Initialize randomly and solve for multiple intervals
        # The IC is discarded later
        Y0 = np.random.rand(nStates).reshape(1,-1)*2*np.pi
        trj, inp, ccs = [Y0], [], []
        for i in range(nStint):
            sol = solve_ivp(kuramotoODE,t_span=(t0,t1),
                            y0=trj[-1][-1],
                            args=(W, AAs[Aidx[j][i]]),
                            t_eval=T+dt, method='Radau')
            trj.append(sol.y.T)      # (nTime, nStates)
            inp.append(np.ones((len(T), nSubSys)) * Aidx[j][i])
            ccs.append(np.tile(CCs[Aidx[j][i]], (len(T), 1, 1)))
        trj = np.vstack(trj[1:])     # (nStint*nTime, nStates)
        inp = np.vstack(inp)         # (nStint*nTime, nStates)
        ccs = np.vstack(ccs)         # (nStint*nTime, nSubSys, nSubSys)

        phi = trj.reshape(-1, nSubSys, nNodes)
        tmp = np.concatenate([np.cos(phi), np.sin(phi)], axis=-1)
        obs = np.mean(tmp, axis=-1)  # (nStint*nTime, nSubSys)

        xs.append(obs)
        ys.append(tmp.reshape(-1, nSubSys*nNodes*2))
        us.append(inp)
        adj.append(ccs)

    _n = int(Ntrj * split)
    np.savez_compressed(
        f'./data/data{suf}_train.npz', x=xs[:_n], y=ys[:_n], u=us[:_n], adj=adj[:_n], Aidx=Aidx[:_n])
    np.savez_compressed(
        f'./data/data{suf}_test.npz', x=xs[_n:], y=ys[_n:], u=us[_n:], adj=adj[_n:], Aidx=Aidx[_n:])

if ifplt:
    data = np.load(f'./data/data{suf}_train.npz', allow_pickle=True)
    xs = data['x']
    us = data['u']
    adj = data['adj']
    Aidx = data['Aidx']

    Ntrj = len(xs)
    Nt, Nx = xs[0].shape
    T = np.arange(Nt)

    f, ax = plt.subplots(Nx//2+1, 2, sharex=True, figsize=(8,Nx//2+1))
    ax = ax.flatten()
    tdx = (Nt//nStint) * np.arange(1, nStint)
    for n in range(Nx):
        for t in tdx:
            ax[n].axvline(t, color='k', linestyle='--')
    for i in range(Ntrj):
        for n in range(Nx):
            ax[n].plot(T, xs[i][:,n], label=f'SubSys {n+1}')
            ax[n].set_ylabel(f'State {n+1}')
        ax[-1].plot(T, us[i][:,0])
        ax[-1].set_ylabel('Topology')

    plt.tight_layout()
    # plt.savefig(f'./data/trajectory{suf}.png')
    plt.show()
