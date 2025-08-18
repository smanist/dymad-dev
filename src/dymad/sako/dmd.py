"""
@package dmd

The implementation of DMD algorithms, including
+ Vanilla EDMD - with linear observables it becomes DMD
+ ResDMD by Colbrook et al.
+ Kernel extensions of the above.

@author Dr. Daning Huang
@date 06/20/23
"""
import numpy as np
from .base import DMDBase
from .sako import SAKO
from .utils import scaledEig, truncatedSVD

# ----------------------------------------------------------------------------------
class EDMD(DMDBase):
    """
    Standard EDMD where the least-squares is solved by truncated SVD.
    """
    def __init__(self, **kwargs):
        """
        @param order: If a float - level of energy to trim the SVD
        """
        _type = kwargs.pop('type', 'EDMD')
        super().__init__(**kwargs)
        self._type = _type

    def _solve_ls(self):
        """
        Solve the least-squares by truncated SVD
        """
        _Ur, _Sr, _Vr = truncatedSVD(self._P0, self._order)
        self._Nrank = len(_Sr)

        _B = (_Ur.conj().T / _Sr.reshape(-1,1)).dot(self._P1)
        _At = _B.dot(_Vr)
        self._wd, _vl, _vr = scaledEig(_At)
        self._vl = _B.conj().T.dot(_vl) / self._wd.conj().reshape(1,-1)
        self._vr = _Vr.dot(_vr)

        # For data member consistency
        self._wd_full = self._wd
        self._vl_full = self._vl
        self._vr_full = self._vr

# ----------------------------------------------------------------------------------
class ResDMD(DMDBase):
    """
    ResDMD implementation - non-kernel version.
    """
    def __init__(self, **kwargs):
        """
        @param order: If a float - threshold of residual to trim eigenpairs.
                      If an integer - number of eigenpairs of minimal residuals to keep.
        @param pord: Order of projection; default to None for no projection.
        @param reps: Threshold for residual.  Default 1e-10
        """
        self._reps = kwargs.pop('reps', 1e-10)
        self._pord = kwargs.pop('pord', None)
        _type = kwargs.pop('type', 'ResDMD')
        super().__init__(**kwargs)
        self._type = _type

    def resolvent_analysis(self, z, return_vec, mode, method):
        """
        Perform resolvent analysis of the DMD operator.

        @param method: 'standard' - See `DMDBase`.
                       'sako' - Use the pseudospectrum functionalities from SAKO
        """
        _ifcont = mode.lower() == 'cont'
        if method.lower() == 'sako':
            if _ifcont:
                # In continuous mode, the inquiry point will be on continuous complex plane
                # But the SAKO formulation is always for discrete time.
                _z = np.exp(z*self._dt)
            else:
                _z = z
            if return_vec:
                _e, _v = self._sako._ps_point(_z, True)
                # _v is the output mode, then recover the input mode by
                # u=(K-zI)v
                _b  = self._proj.dot(_v)
                _ls = self._wd.conj().reshape(-1,1)
                _u = (self._vl*_b).dot(_ls).reshape(-1)
                _u -= _z*_v
            else:
                _e = self._sako._ps_point(_z, False)
            if _ifcont:
                # The gain is in discrete time, and we convert it back
                _e *= self._dt
            if return_vec:
                return _e, _v, _u
            return _e
        # Otherwise we use the method from base class
        return super().resolvent_analysis(z, return_vec, mode, method)

    def _solve_ls(self):
        """
        Solve a generalized eigenvalue problem to obtain the eigensystem of the full-order
        operator A from the least-squares, and then filter the resulting eigenpairs by the residual.
        """
        ## Create the SAKO object
        if self._pord is None:
            self._sako = SAKO(self._P0, self._P1, self._data._W, dt=self._dt, reps=self._reps, verbose=self._verbose)
        else:
            self._sako = SAKOproj(self._P0, self._P1, self._data._W, dt=self._dt, reps=self._reps, order=self._pord,
                                  verbose=self._verbose)
        _M0 = self._sako._M0
        _M1 = self._sako._M1
        ## Generalized eigenvalue problem
        self._wd_full, _vl_full, self._vr_full = scaledEig(_M1, _M0)
        self._vl_full = _M0.conj().T.dot(_vl_full)

        ## Filtering of ResDMD
        self.filter_spectrum(self._sako, self._order)
