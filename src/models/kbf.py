import jax.numpy as jn
import jax
import optax
import numpy as np

from ..utils import init_mat_kaiming as init_mat
from .model import ModelBase

class KBF_ENC(ModelBase):
    """Full autoencoder form for KBF."""
    def __init__(self, dims, nums, ifone, act):
        self.Nx, self.Nu, self.Nk = dims
        self.Nns   = np.atleast_1d(nums)
        self.Nl    = len(nums)+1
        self.ifone = ifone
        self.act   = act

        if ifone:
            self.encoder = self._encoder_one
        else:
            self.encoder = self._encoder_smpl

    def init_params(self) -> optax.Params:
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        dec = np.array(enc[::-1])
        if self.ifone:
            enc[-1] -= 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2], _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1],), _i+200)})
            _p.update({f'de{_i}': init_mat(dec[_i:_i+2], _i+100)})
            _p.update({f'db{_i}': init_mat((dec[_i+1],), _i+300)})
        _p.update(As = init_mat([self.Nk*(self.Nu+1), self.Nk], 415411).T)

        return _p

    def _encoder_smpl(self, x: jn.ndarray, params: optax.Params) -> jn.ndarray:
        for _i in range(self.Nl-1):
            x = jn.dot(x, params[f'en{_i}']) + params[f'eb{_i}']
            x = self.act(x)
        x = jn.dot(x, params[f'en{self.Nl-1}']) + params[f'eb{self.Nl-1}']
        return x

    def _encoder_one(self, x: jn.ndarray, params: optax.Params) -> jn.ndarray:
        x = jn.atleast_2d(self._encoder_smpl(x, params))
        return jn.hstack([jn.ones((len(x),1)), x]).squeeze()

    def decoder(self, z: jn.ndarray, params: optax.Params) -> jn.ndarray:
        for _i in range(self.Nl-1):
            z = jn.dot(z, params[f'de{_i}']) + params[f'db{_i}']
            z = self.act(z)
        z = jn.dot(z, params[f'de{self.Nl-1}']) + params[f'db{self.Nl-1}']
        return z

    def features(self, x: jn.ndarray, u: jn.ndarray, params: optax.Params) -> jn.ndarray:
        z = self.encoder(x, params)
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        return f

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: optax.Params) -> jn.ndarray:
        """For prediction."""
        w = jn.hstack([[1], u]).reshape(-1,1)
        f = (w*z).reshape(-1)
        dz = jn.dot(params['As'], f)
        return dz

    def predict(self, x0: jn.ndarray, us: jn.ndarray, params: optax.Params) -> jn.ndarray:
        Nt = len(us)
        zp, xp = [self.encoder(x0, params)], [x0]
        for _i in range(1,Nt):
            zp.append( self.dynamics(zp[-1], us[_i-1], params) )
            xp.append( self.decoder(zp[-1], params) )
        return jn.vstack(zp), jn.vstack(xp)

class KBF_LND(KBF_ENC):
    """Linear decoder case."""
    def init_params(self) -> optax.Params:
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        if self.ifone:
            enc[-1] -= 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2], _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1],), _i+200)})
        _p.update(de = init_mat([self.Nk, self.Nx], 100))
        _p.update(As = init_mat([self.Nk*(self.Nu+1), self.Nk], 415411).T)

        return _p

    def decoder(self, z: jn.ndarray, params: optax.Params) -> jn.ndarray:
        z = jn.dot(z, params['de'])
        return z

class KBF_AUT(KBF_ENC):
    """
    Full autoencoder form for autonomous dynamics.
    Simplified from KBF_ENC, and the interface is maintained the same.
    """
    def __init__(self, dims, nums, ifone, act):
        tmp = [dims[0], 0, dims[1]]
        super().__init__(tmp, nums, ifone, act)

    def init_params(self) -> optax.Params:
        _p = super().init_params()
        _p.update(As = init_mat([self.Nk, self.Nk], 415411))
        return _p

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: optax.Params) -> jn.ndarray:
        dz = jn.dot(params['As'], z)
        return dz

class KBF_SMPL(KBF_ENC):
    def __init__(self, dims):
        self.Nx, self.Nu, self.Nk = dims

    def init_params(self) -> optax.Params:
        _p = {}
        _p.update(As = init_mat([self.Nk*(self.Nu+1), self.Nk], 415411).T)
        return _p

    def encoder(self, x: jn.ndarray, params: optax.Params) -> jn.ndarray:
        _x = jn.atleast_2d(x)
        Nt = len(_x)
        _z = jn.hstack([np.ones((Nt,1)), _x, _x[:,0].reshape(-1,1)**2])
        return _z.squeeze()

    def decoder(self, z: jn.ndarray, params: optax.Params) -> jn.ndarray:
        _z = jn.atleast_2d(z)
        _x = _z[:,1:3]
        return _x.squeeze()

class KBF_STK(KBF_ENC):
    """Stacked encoder case.  Always assuming 1 on top."""
    def init_params(self) -> optax.Params:
        enc = np.hstack([[self.Nx], self.Nns, [self.Nk]])
        enc[-1] -= 1  # Saving for 1

        _p = {}
        for _i in range(self.Nl):
            _p.update({f'en{_i}': init_mat(enc[_i:_i+2], _i)})
            _p.update({f'eb{_i}': init_mat((enc[_i+1],), _i+200)})
        _p.update(As = init_mat([self.Nk*(self.Nu+1), self.Nk], 415411).T)

        return _p

    def encoder(self, x: jn.ndarray, params: optax.Params) -> jn.ndarray:
        _x = jn.atleast_2d(x)
        for _i in range(self.Nl-1):
            x = jn.dot(x, params[f'en{_i}']) + params[f'eb{_i}']
            x = self.act(x)
        x = jn.dot(x, params[f'en{self.Nl-1}']) + params[f'eb{self.Nl-1}']
        Nt = len(_x)
        _z = jn.hstack([np.ones((Nt,1)), _x, x])
        return _z.squeeze()

    def decoder(self, z: jn.ndarray, params: optax.Params) -> jn.ndarray:
        return z[1:self.Nx+1]
