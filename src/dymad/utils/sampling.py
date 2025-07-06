import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import chirp
from typing import Callable, Dict, Tuple, Union
import yaml

Array = np.ndarray

logger = logging.getLogger(__name__)

# -----------------------
# Control Samplers
# -----------------------
# Basic samplers
def chirp_control(*,
                  t1:           float,
                  dim:          int,
                  freq_range:   Tuple[float,float]  = (0.1, 2.0),
                  amp_range:    Tuple[float,float]  = (0.5, 1.0),
                  phase_range:  Tuple[float,float]  = (0.0, 360.),    # In deg
                  method:       str = "linear",
                  rng:          np.random.Generator = None) -> Callable:
    f0, f1 = freq_range
    A = rng.uniform(*amp_range)
    P = rng.uniform(*phase_range)
    amplitude = np.broadcast_to(A, (dim,))
    def _sampler(t_grid: Array, i: int) -> Array:
        base = chirp(t_grid, f0=f0, f1=f1, t1=t1,
                     method=method, phi=P)
        if isinstance(t_grid, float):
            return base * amplitude
        else:
            return base[:, None] * amplitude
    return _sampler

def gaussian_control(*,
                     mean: Union[float, Array],
                     std:  Union[float, Array],
                     t1:   float,
                     dt:   float,
                     dim:  int,
                     mode: str = "zoh",
                     rng:  np.random.Generator = None) -> Callable:
    mean = np.broadcast_to(mean, (dim,))
    std  = np.broadcast_to(std,  (dim,))
    Nt   = int(np.ceil(t1 / dt)) + 1
    ts   = np.arange(Nt) * dt
    us   = rng.normal(mean, std, size=(ts.size, dim))
    _int = _build_interpolant(ts, us, mode)
    def _sampler(t_grid: Union[float, Array], i: int, _rng=rng) -> Array:
        return _int(t_grid)
    return _sampler

def sine_control(*,
                 dim:            int,
                 num_components: int                 = 1,
                 freq_range:     Tuple[float,float]  = (0.1, 2.0),
                 amp_range:      Tuple[float,float]  = (0.5, 1.0),
                 phase_range:    Tuple[float,float]  = (0.0, 360.),    # In deg,
                 rng:            np.random.Generator = None) -> Callable:
    A = rng.uniform(*amp_range,   size=(dim, num_components))
    F = rng.uniform(*freq_range,  size=(dim, num_components))
    P = rng.uniform(*phase_range, size=(dim, num_components))/180*np.pi
    def _sampler(t_grid: Union[float, Array], i: int) -> Array:
        if isinstance(t_grid, float):
            return np.sum(A*np.sin(2*np.pi*F*t_grid + P))
        else:
            t = t_grid[:, None, None]
            return np.sum(A*np.sin(2*np.pi*F*t + P), axis=(1,2))
    return _sampler

def sphere_control(*,
                   radius: Union[float, Array],
                   t1:   float,
                   dt:   float,
                   dim:  int,
                   mode: str = "zoh",
                   rng:  np.random.Generator = None) -> Callable:
    rad  = np.broadcast_to(radius, (dim,))
    Nt   = int(np.ceil(t1 / dt)) + 1
    ts   = np.arange(Nt) * dt
    us   = rng.normal(0, 1, size=(ts.size, dim))
    us  /= np.maximum(np.linalg.norm(us, axis=1, keepdims=True), 1e-15)
    us  *= rad
    _int = _build_interpolant(ts, us, mode)
    def _sampler(t_grid: Union[float, Array], i: int, _rng=rng) -> Array:
        return _int(t_grid)
    return _sampler

# Collection of samplers
_CTRL_MAP = {
    "chirp"    : chirp_control,
    "gaussian" : gaussian_control,
    "sine"     : sine_control,
    "sphere"   : sphere_control,
}

# Helper function
def _build_interpolant(t: Array, u: Array, mode: str) -> Callable:
    """Return callable u(t_query) according to interpolation mode."""
    mode = mode.lower()
    if mode == "zoh":
        def _u(tq):
            idx = np.searchsorted(t, tq, side="right") - 1
            idx = np.clip(idx, 0, len(t)-1)
            return u[idx]
        return _u

    if mode == "linear":
        interp = interp1d(t, u, axis=0, bounds_error=False,
                            fill_value="extrapolate", assume_sorted=True)
        return lambda tq: interp(tq).astype(float)

    if mode == "cubic":
        cs = CubicSpline(t, u, axis=0, bc_type="natural", extrapolate=True)
        return lambda tq: cs(tq).astype(float)

    raise ValueError(f"Unknown interpolation mode '{mode}'.")

# -----------------------
# Initial Condition Samplers
# -----------------------

def gaussian_x0(*,
                mean: Union[float, Array],
                std:  Union[float, Array],
                dim:  int,
                rng:  np.random.Generator = None) -> Callable:
    mean = np.broadcast_to(mean, (dim,))
    std  = np.broadcast_to(std,  (dim,))
    def _sampler(i: int, _rng=rng) -> Array:
        return _rng.normal(mean, std, size=(dim,))
    return _sampler

def grid_x0(*,
            bounds: Union[float, Array],
            dim:  int,
            n_points: int = 3,
            rng:  np.random.Generator = None) -> Callable:
    bounds = np.broadcast_to(bounds, (dim,2))
    n_points = np.broadcast_to(n_points, (dim,))
    xs   = [np.linspace(bounds[i,0], bounds[i,1], n_points[i]) for i in range(dim)]
    msh  = np.meshgrid(*xs, indexing='ij')
    arr  = np.stack(msh, axis=-1).reshape(-1, dim)
    def _sampler(i: int, _arr=arr) -> Array:
        return _arr[i]
    return _sampler

def uniform_x0(*,
               bounds: Union[float, Array],
               dim:  int,
               rng:  np.random.Generator = None) -> Callable:
    bounds = np.broadcast_to(bounds, (dim,2)).T
    def _sampler(i: int, _rng=rng) -> Array:
        return _rng.uniform(low=bounds[0], high=bounds[1], size=(dim,))
    return _sampler

_X0_MAP = {
    "gaussian" : gaussian_x0,
    "grid"     : grid_x0,
    "uniform"  : uniform_x0,
}

# -----------------------
# Trajectory Samplers
# -----------------------
class TrajectorySampler:
    r"""
    Generate batches of trajectories for
        \dot{x} = f(t, x, u)
        y = g(t, x, u)
    """
    def __init__(self,
                 f: Callable[[float, Array, Array], Array],
                 g: Callable[[float, Array, Array], Array] = None,
                 config: Union[str, Dict] = None,
                 rng: Union[int, np.random.Generator, None] = None,
                 config_mod: Dict = None):
        self.f   = f
        self.g   = (lambda t, x, u: x) if g is None else g
        self.rng = np.random.default_rng(rng)

        with open(config, "r") as f:
            self.config = yaml.safe_load(f)
        if config_mod is not None:
            if not isinstance(config_mod, dict):
                raise TypeError("config_mod must be a dictionary.")
            self.config.update(config_mod)

        tmp = self.config.get("dims", None)
        if tmp is None:
            raise ValueError("Config must specify 'dims' (state/observation/input dimensions).")
        self.dims = [tmp["states"], tmp["inputs"], tmp["observations"]]

        logger.info(f"TrajectorySampler initialized with dims: "
                    f"states={self.dims[0]}, inputs={self.dims[1]}, "
                    f"observations={self.dims[2]}")
        logger.info(f"Control config: {self.config.get('control', None)}")
        logger.info(f"Init. Cond. config: {self.config.get('x0', None)}")
        logger.info(f"Solver config: {self.config.get('solver', None)}")

    def _create_control_sampler(self,
                                t_grid: Array,
                                traj_idx: int,
                                ) -> Tuple[Callable[[float], Array], Array]:
        """
        Returns (u_callable, u_grid).  Choice depends on u_spec.
        Supported modes: 'zoh', 'linear', 'cubic'.
        """
        u_spec = self.config.get("control", None)
        if u_spec is None:
            # Autonomous
            return lambda t: np.zeros_like(t), np.zeros((t_grid.size,))

        if callable(u_spec):
            # Externally supplied function
            u_call = lambda t, idx=traj_idx: np.asarray(u_spec(t, idx))
            u_grid = np.stack([u_call(t) for t in t_grid])
            return u_call, u_grid

        if isinstance(u_spec, Array):
            # Externally supplied array data
            U_vec = u_spec if u_spec.ndim == 2 else u_spec[traj_idx]
            assert t_grid.size == U_vec.shape[0], \
                f"t_grid size {t_grid.size} does not match U_vec size {U_vec.shape[0]}"
            mode = u_spec.get("mode", "cubic").lower()
            u_call = _build_interpolant(t_grid, U_vec, mode)
            return u_call, U_vec

        if isinstance(u_spec, dict):
            # Defined by a dictionary
            kind = u_spec["kind"].lower()
            if kind not in _CTRL_MAP:
                raise KeyError(f"Unknown control kind '{kind}'. Available: {list(_CTRL_MAP)}")
            params = u_spec.get("params", {})
            params.update({
                "dim": self.dims[1],
                "rng": self.rng})
            u_func = _CTRL_MAP[kind](**params)
            u_call = lambda t, i=traj_idx: u_func(t, i)
            U_vec = u_call(t_grid)
            return u_call, U_vec.reshape(-1, self.dims[1])

        raise TypeError("Unrecognised u_spec type.")

    def _sample_x0(self, traj_idx: int) -> Array:
        x0_spec = self.config.get("x0", None)

        if isinstance(x0_spec, Array):
            # Externally supplied array data
            x0_arr = np.asarray(x0_spec)
            return x0_arr if x0_arr.ndim == 1 else x0_arr[traj_idx]

        if isinstance(x0_spec, dict):
            # Defined by a dictionary
            kind = x0_spec["kind"].lower()
            if kind not in _X0_MAP:
                raise KeyError(f"Unknown x0 kind '{kind}'. Available: {list(_X0_MAP)}")
            params = x0_spec.get("params", {})
            params.update({
                "dim": self.dims[0],
                "rng": self.rng})
            x0_func = _X0_MAP[kind](**params)
            return np.asarray(x0_func(traj_idx))

        raise TypeError("Unrecognised x0_spec type.")

    def sample(self,
               t_samples: Array,
               batch: int = 1) -> dict:
        """
        Returns dict with shapes
            t : (B,T)
            u : (B,T,k)
            x : (B,T,n)
            y : (B,T,m)
        """
        tt = np.asarray(t_samples)
        Nt = tt.size
        opts = self.config.get("solver", {})

        ts = np.zeros((batch, Nt))
        xs = np.zeros((batch, Nt, self.dims[0]))
        us = np.zeros((batch, Nt, self.dims[1]))
        ys = np.zeros((batch, Nt, self.dims[2]))

        for i in range(batch):
            logger.info(f"Generating trajectory {i+1}/{batch}...")

            fu, uu = self._create_control_sampler(tt, i)
            def rhs(t, x):
                return self.f(t, x, fu(t))

            x0  = self._sample_x0(i)
            sol = solve_ivp(rhs,
                            (tt[0], tt[-1]),
                            x0,
                            t_eval=tt,
                            **opts)
            if not sol.success:
                raise RuntimeError(f"Integration failed on traj {i}: {sol.message}")
            xx = sol.y.T
            yy = self.g(tt, xx, uu)

            ts[i], xs[i], us[i], ys[i] = tt, xx, uu, yy

        return ts, xs, us, ys
