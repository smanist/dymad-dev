
# ---------------------
# Lift
# ---------------------
## Single state
def psiFourier(x, K):
    _o = K % 2
    _p = (K+1) // 2
    return np.exp(1j*x.reshape(-1,1) * (_p*(-1)**_o))

def psiFourierReal(x, K):
    _x = x.reshape(-1)
    _p = [np.ones_like(_x)]
    for _k in K:
        _p.append(np.cos((_k+1)*_x))
        _p.append(np.sin((_k+1)*_x))
    return np.vstack(_p).T

def psiMonomial(x, K):
    return x.reshape(-1,1)**K

## Cross-product of psi's
def psiCross(x, ps, Ks):
    _Nx = len(x)
    _Np = len(ps)
    _p1 = ps[0](x[:,0], np.arange(Ks[0]))
    for _i in range(1, _Np):
        _p2 = ps[_i](x[:,_i], np.arange(Ks[_i]))
        _p1 = np.einsum('ij,ik->ijk', _p1, _p2).reshape(_Nx, -1)
    return _p1

class Lift(Transform):
    """
    Lifting into higher-dimensional space.
    """
    def __init__(self, Xs, **kwargs):
        """
        @param fobs: Lifting function from input dim M to output dim L for N steps.
                     input: NxM array; output: NxL array
        @param finv: list of integers: Indices of outputs to return
                     'pinv': Pseudo-inverse from outputs
        """
        super().__init__(Xs, **kwargs)
        self._fobs = kwargs.pop('fobs')
        self._rcond = kwargs.pop('rcond', 1e-15)
        self._Nout = self._fobs(Xs[0][0].reshape(1,-1)).shape[1]  # Probe the output dimension

    def __str__(self):
        return "lift"

    def fit(self, X: Array) -> None:
        """"""

        self.proc_data(X)

        self._C = None
        finv = kwargs.pop('finv')
        if isinstance(finv, list):
            self._C = np.zeros((self._Nout, self._Ninp))
            for _i, _f in enumerate(finv):
                self._C[_f,_i] = 1.0
        elif isinstance(finv, np.ndarray):
            self._C = np.copy(finv)
        elif finv == 'pinv':
            _Z = np.vstack(self._Zs)
            self._C = np.linalg.pinv(_Z, rcond=self._rcond, hermitian=False).dot(np.vstack(Xs))
        elif callable(finv):
            self._finv = finv
        else:
            raise ValueError(f"Unknown finv option {finv}")
        if self._C is None:
            self.DE = self._decoder_inv
        else:
            self.DE = self._decoder_lin

    def transform(self, X: Array) -> Array:
        return self._fobs(X.reshape(-1,self._Ninp))

    def inverse_transform(self, X: Array) -> Array:
        pass

    def _decoder_lin(self, Z):
        return Z.dot(self._C)

    def _decoder_inv(self, Z):
        return self._finv(Z.reshape(-1,self._Nout))
