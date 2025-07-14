import logging
import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import chirp
from typing import Callable, Dict, Tuple, Union

from dymad.utils.misc import load_config

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
    """
    Generate a chirp control signal.

    See `scipy.signal.chirp` for technical details.

    Args:
        t1 (float): End time of the chirp, which can be shorter than the duration of signal.
        dim (int): Dimension of the control signal.
        freq_range (Tuple[float, float]): Frequency range (f0, f1) in Hz.
        amp_range (Tuple[float, float]): Amplitude range (min, max).
        phase_range (Tuple[float, float]): Phase range (min, max) in degrees.  Default is (0, 360).
        method (str): Method for temporal variation in frequency.
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes a time grid and returns the chirp signal.
    """
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
    """
    Generate a Gaussian control signal.

    Args:
        mean (Union[float, Array]): Mean of the Gaussian distribution.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,).

        std (Union[float, Array]): Standard deviation of the Gaussian distribution.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,).

        t1 (float): End time of the Gaussian signal.
        dt (float): Time step for the Gaussian signal.
        dim (int): Dimension of the control signal.
        mode (str): Interpolation mode ('zoh', 'linear', 'cubic').
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes a time grid and returns the Gaussian signal.
    """
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
                 freq_range:     Tuple[float,float]  = (0.1, 2.0),     # In Hz,
                 amp_range:      Tuple[float,float]  = (0.5, 1.0),
                 phase_range:    Tuple[float,float]  = (0.0, 360.),    # In deg,
                 rng:            np.random.Generator = None) -> Callable:
    """
    Generate a sine control signal with multiple components.

    Args:
        dim (int): Dimension of the control signal.
        num_components (int): Number of sine components.
        freq_range (Tuple[float, float]): Frequency range (f_min, f_max) in Hz.
        amp_range (Tuple[float, float]): Amplitude range (min, max).
        phase_range (Tuple[float, float]): Phase range (min, max) in degrees.  Default is (0, 360).
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes a time grid and returns the sine signal.
    """
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
    """
    Generate a control signal on the surface of a sphere.

    Args:
        radius (Union[float, Array]): Radius of the sphere.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,).

        t1 (float): End time of the control signal.
        dt (float): Time step for the control signal.
        dim (int): Dimension of the control signal.
        mode (str): Interpolation mode ('zoh', 'linear', 'cubic').
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes a time grid and returns the control signal on the sphere.
    """
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
    """
    Generate a Gaussian initial condition sampler.

    Args:
        mean (Union[float, Array]): Mean of the Gaussian distribution.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,).

        std (Union[float, Array]): Standard deviation of the Gaussian distribution.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,).

        dim (int): Dimension of the initial condition.
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that returns a sample from the Gaussian distribution.
    """
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
    """
    Generate a grid-based initial condition sampler.

    Args:
        bounds (Union[float, Array]): Bounds for the grid sampling.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,2).

        dim (int): Dimension of the initial condition.
        n_points (int): Number of points in the grid for each dimension.
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes an index and returns a sample from the grid-based initial condition.
    """
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
    """
    Generate a uniformly random initial condition sampler.

    Args:
        bounds (Union[float, Array]): Bounds for the uniform sampling.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,2).

        dim (int): Dimension of the initial condition.
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes an index and returns a sample from the uniform distribution.
    """
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
    r"""Sampler for generating trajectories.

    This class generates batches of trajectories based on a system defined by
    the functions `f` and `g`, which represent the system dynamics and observation model,
    respectively. The trajectories are sampled according to the configuration specified
    in the provided YAML file or dictionary.

    The dynamics are

    .. math::
        \begin{align*}
        \dot{x} &= f(t, x, u) \\
        y &= g(t, x, u)
        \end{align*}

    Args:
        f (Callable[[float, Array, Array], Array]): Function defining the system dynamics.
            It should take time `t`, state `x`, and control input `u` as arguments and return the state derivative.
        g (Callable[[float, Array, Array], Array], optional): Function defining the observation
            model. It should take time `t`, state `x`, and control input `u` as arguments and return the observation.
            If not provided, it defaults to the identity function (`g(t, x, u) = x`).
        config (Union[str, Dict], optional): Path to a YAML configuration file or a dictionary
            containing the configuration for the sampler. The configuration should specify the dimensions
            of the states, inputs, and observations, as well as control and initial condition specifications.
        rng (Union[int, np.random.Generator, None], optional): Random number generator or seed for reproducibility.
            If an integer is provided, it is used to seed the default random number generator.
            If `None`, the default random generator is used.
        config_mod (Dict, optional): Additional configuration parameters to modify the loaded configuration.
            This should be a dictionary that updates or overrides the values in the loaded configuration.
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

        self.config = load_config(config, config_mod)

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
               batch: int = 1,
               save = None) -> dict:
        """
        Sample trajectories for a given time grid.

        Args:
            t_samples (Array): Time samples for the trajectory.
                Should be a 1D array of time points.  Assuming length T.
            batch (int): Number of trajectories to sample.  Assuming B trajectories.
            save (str, optional): If provided, saves the sampled trajectories to a file.
                The states are discarded.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A dictionary containing the sampled trajectories. The keys are:

                - 't': Time samples (shape: (B, T))
                - 'u': Control inputs (shape: (B, T, k))
                - 'x': States (shape: (B, T, n))
                - 'y': Observations (shape: (B, T, m))
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

        if save is not None:
            assert isinstance(save, str), "Save path must be a string."
            os.makedirs(os.path.dirname(save), exist_ok=True)
            np.savez_compressed(save, t=ts, x=ys, u=us)

        return ts, xs, us, ys