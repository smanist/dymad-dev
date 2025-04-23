from typing import Callable, Union
import jax.numpy as jn
import jax
import optax
import numpy as np
import scipy.integrate as spi

class ModelBase:
    """
    Notation:
    x: Physical space
    z: Embedding space
    Both are treated row-wise, to be consistent with training procedure.

    Discrete-time model:
    z_k = encoder(x_k)
    z_k+1 = dynamics(z_k, u_k)
    x_k+1 = decoder(z_k+1)

    Continuous-time model:
    z = encoder(x)
    z_dot = dynamics(z, u)
    x = decoder(z)
    """
    def init_params(self) -> optax.Params:
        """
        Initialize the dict of model parameters.
        """
        raise NotImplementedError("This is the base class.")

    def encoder(self, x: jn.ndarray, params: optax.Params) -> jn.ndarray:
        """Map x states to z states."""
        raise NotImplementedError("This is the base class.")

    def decoder(self, z: jn.ndarray, params: optax.Params) -> jn.ndarray:
        """Map z states to x states."""
        raise NotImplementedError("This is the base class.")

    def features(self, x: jn.ndarray, u: jn.ndarray, params: optax.Params) -> jn.ndarray:
        """
        If the dynamics can be written as a linear combination of nonlinear features,
        this function computes these nonlinear features.
        This is needed for the two-step optimization.
        """
        raise NotImplementedError("This is the base class.")

    def dynamics(self, z: jn.ndarray, u: jn.ndarray, params: optax.Params) -> jn.ndarray:
        """Advance one step in z space."""
        raise NotImplementedError("This is the base class.")

    def predict(self, x0: jn.ndarray, us: jn.ndarray, params: optax.Params) -> Union[jn.ndarray,jn.ndarray]:
        """
        Prediction in DT mode.

        Args:
        x0: Initial conditions in x space.
        us: List of step inputs.
        params: Model parameters.

        Returns:
        Solutions in x and z spaces.
        """
        raise NotImplementedError("This is the base class.")

def predict_ct(
    model: ModelBase,
    params: optax.Params,
    x0: jn.ndarray,
    ts: jn.ndarray,
    ut: Callable,
    ifproj: bool = False):
    if ifproj:
        def f(t, z):
            z = model.encoder(model.decoder(z, params), params)
            dz = model.dynamics(z, ut(t), params)
            return np.array(dz)
    else:
        def f(t, z):
            dz = model.dynamics(z, ut(t), params)
            return np.array(dz)
    z0  = model.encoder(x0, params)
    sol = spi.solve_ivp(f, [0, ts[-1]], z0, t_eval=ts, method='RK45')
    xs  = model.decoder(sol.y.T, params)
    return np.array(xs)

def predict_dt(
    model: ModelBase,
    params: optax.Params,
    x0: jn.ndarray,
    us: jn.ndarray):
    zs, xs = model.predict(x0, us, params)
    return np.array(xs)

def predict_ct_dt(
    model: ModelBase,
    params: optax.Params,
    x0: jn.ndarray,
    dt, float,
    ut: jn.ndarray,
    ifproj: bool = False):
    Nt = len(ut)
    xs = [x0]
    for _i in range(Nt):
        _x = xs[-1]
        _u = lambda t: ut[_i]
        sol = predict_ct(model, params, _x, [0, dt], _u, ifproj)
        xs.append(sol[-1])
    return np.array(xs)