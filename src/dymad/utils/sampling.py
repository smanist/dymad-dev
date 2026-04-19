import copy
import hashlib
import logging
import os
import pickle
from collections.abc import Callable
from os import PathLike
from typing import Any, cast

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import chirp

from dymad.utils.misc import load_config

Array = np.ndarray
Rng = np.random.Generator | None
ConfigLike = str | PathLike[str] | dict[str, Any] | None
ControlSampler = Callable[[float | Array, int], Array]
StateSampler = Callable[[int], Array]
NoiseSampler = Callable[[Array, int], Array]
SampleWithControl = tuple[Array, Array, Array, Array] | tuple[Array, Array, Array, Array, Array]
SampleAutonomous = tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]


def _require_rng(rng: Rng) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _independent_rng(
    rng: int | np.random.Generator | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        state_bytes = pickle.dumps(rng.bit_generator.state, protocol=5)
        digest = hashlib.blake2b(state_bytes, digest_size=16).digest()
        entropy = np.frombuffer(digest, dtype=np.uint32)
        return np.random.default_rng(np.random.SeedSequence(entropy))
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(rng), 1]))


logger = logging.getLogger(__name__)


# -----------------------
# Control Samplers
# -----------------------
# Basic samplers
def chirp_control(
    *,
    t1: float,
    dim: int,
    freq_range: tuple[float, float] = (0.1, 2.0),
    amp_range: tuple[float, float] = (0.5, 1.0),
    phase_range: tuple[float, float] = (0.0, 360.0),  # In deg
    method: str = "linear",
    rng: Rng = None,
) -> ControlSampler:
    """
    Generate a chirp control signal.

    See `scipy.signal.chirp` for technical details.

    Args:
        t1 (float): The time at which f1 is specified, which can be shorter than the duration of signal.
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
    rng = _require_rng(rng)
    f0, f1 = freq_range
    A = rng.uniform(*amp_range)
    P = rng.uniform(*phase_range)
    amplitude = np.broadcast_to(A, (dim,))

    def _sampler(t_grid: float | Array, i: int) -> Array:
        del i
        base = chirp(t_grid, f0=f0, f1=f1, t1=t1, method=method, phi=int(P))
        if isinstance(t_grid, float):
            return base * amplitude
        return cast(Array, base[:, None] * amplitude)

    return _sampler


def gaussian_control(
    *,
    mean: float | Array,
    std: float | Array,
    t1: float,
    dt: float,
    dim: int,
    mode: str = "zoh",
    rng: Rng = None,
) -> ControlSampler:
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
    rng = _require_rng(rng)
    mean = np.broadcast_to(mean, (dim,))
    std = np.broadcast_to(std, (dim,))
    Nt = int(np.ceil(t1 / dt)) + 1
    ts = np.arange(Nt) * dt
    us = rng.normal(mean, std, size=(ts.size, dim))
    _int = _build_interpolant(ts, us, mode)

    def _sampler(t_grid: float | Array, i: int) -> Array:
        del i
        return _int(t_grid)

    return _sampler


def sine_control(
    *,
    dim: int,
    num_components: int = 1,
    freq_range: tuple[float, float] = (0.1, 2.0),  # In Hz,
    amp_range: tuple[float, float] = (0.5, 1.0),
    phase_range: tuple[float, float] = (0.0, 360.0),  # In deg,
    rng: Rng = None,
) -> ControlSampler:
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
    rng = _require_rng(rng)
    A = rng.uniform(*amp_range, size=(dim, num_components))
    F = rng.uniform(*freq_range, size=(dim, num_components))
    P = rng.uniform(*phase_range, size=(dim, num_components)) / 180 * np.pi

    def _sampler(t_grid: float | Array, i: int) -> Array:
        del i
        if np.isscalar(t_grid):
            t_scalar = float(cast(Any, t_grid))
            values = np.sum(A * np.sin(2 * np.pi * F * t_scalar + P), axis=1)
            return cast(Array, values.reshape(dim))
        t = np.asarray(t_grid)[:, None, None]
        return cast(Array, np.sum(A * np.sin(2 * np.pi * F * t + P), axis=2))

    return _sampler


def sphere_control(
    *,
    radius: float | Array,
    t1: float,
    dt: float,
    dim: int,
    mode: str = "zoh",
    rng: Rng = None,
) -> ControlSampler:
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
    rng = _require_rng(rng)
    rad = np.broadcast_to(radius, (dim,))
    Nt = int(np.ceil(t1 / dt)) + 1
    ts = np.arange(Nt) * dt
    us = rng.normal(0, 1, size=(ts.size, dim))
    us /= np.maximum(np.linalg.norm(us, axis=1, keepdims=True), 1e-15)
    us *= rad
    _int = _build_interpolant(ts, us, mode)

    def _sampler(t_grid: float | Array, i: int) -> Array:
        del i
        return _int(t_grid)

    return _sampler


#: Mapping of control sampler names to functions.
CTRL_MAP = {
    "chirp": chirp_control,
    "gaussian": gaussian_control,
    "sine": sine_control,
    "sphere": sphere_control,
}


# Helper function
def _build_interpolant(t: Array, u: Array, mode: str) -> Callable[[float | Array], Array]:
    """Return callable u(t_query) according to interpolation mode."""
    mode = mode.lower()
    if mode == "zoh":

        def _u(tq):
            idx = np.searchsorted(t, tq, side="right") - 1
            idx = np.clip(idx, 0, len(t) - 1)
            return u[idx]

        return _u

    if mode == "linear":
        interp = cast(Any, interp1d)(
            t, u, axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True
        )
        return lambda tq: interp(tq).astype(float)

    if mode == "cubic":
        cs = CubicSpline(t, u, axis=0, bc_type="natural", extrapolate=True)
        return lambda tq: cs(tq).astype(float)

    raise ValueError(f"Unknown interpolation mode '{mode}'.")


# -----------------------
# Initial Condition / Parameter Samplers
# -----------------------


def gaussian_x0(
    *,
    mean: float | Array,
    std: float | Array,
    dim: int,
    rng: Rng = None,
) -> StateSampler:
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
    rng = _require_rng(rng)
    mean = np.broadcast_to(mean, (dim,))
    std = np.broadcast_to(std, (dim,))

    def _sampler(i: int) -> Array:
        del i
        return rng.normal(mean, std, size=(dim,))

    return _sampler


def grid_x0(*, bounds: float | Array, dim: int, n_points: int = 3, rng: Rng = None) -> StateSampler:
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
    del rng
    bounds = np.broadcast_to(bounds, (dim, 2))
    n_points_arr = np.broadcast_to(n_points, (dim,))
    xs = [np.linspace(bounds[i, 0], bounds[i, 1], int(n_points_arr[i])) for i in range(dim)]
    msh = np.meshgrid(*xs, indexing="ij")
    arr = np.stack(msh, axis=-1).reshape(-1, dim)

    def _sampler(i: int, _arr=arr) -> Array:
        return _arr[i]

    return _sampler


def uniform_x0(*, bounds: float | Array, dim: int, rng: Rng = None) -> StateSampler:
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
    rng = _require_rng(rng)
    bounds = np.broadcast_to(bounds, (dim, 2)).T

    def _sampler(i: int) -> Array:
        del i
        return rng.uniform(low=bounds[0], high=bounds[1], size=(dim,))

    return _sampler


def perturb_x0(*, bounds: float | Array, dim: int, ref: Array, rng: Rng = None) -> StateSampler:
    """
    Generate a uniformly random initial condition sampler around a reference trajectory.

    Args:
        bounds (Union[float, Array]): Bounds for the uniform sampling.

            - If scalar, it is broadcasted to the dimension.
            - If an array, it should have shape (dim,2).

        dim (int): Dimension of the initial condition.
        ref (Array): Reference trajectory to perturb, shape (n_steps, dim).
        rng (np.random.Generator): Random number generator for sampling.

    Returns:
        Callable:
            A callable that takes an index and returns a perturbed sample.
    """
    rng = _require_rng(rng)
    bounds = np.broadcast_to(bounds, (dim, 2)).T

    def _sampler(i: int) -> Array:
        del i
        _j = rng.integers(0, ref.shape[0])
        return ref[_j] + rng.uniform(low=bounds[0], high=bounds[1], size=(dim,))

    return _sampler


#: Mapping of initial condition sampler names to functions.
X0_MAP = {
    "gaussian": gaussian_x0,
    "grid": grid_x0,
    "perturb": perturb_x0,
    "uniform": uniform_x0,
}


# -----------------------
# Observation Noise Samplers
# -----------------------
def gaussian_noise(
    *,
    mean: float | Array,
    std: float | Array,
    dim: int,
    rng: Rng = None,
) -> NoiseSampler:
    """Generate additive Gaussian observation noise."""
    rng = _require_rng(rng)
    mean = np.broadcast_to(mean, (dim,))
    std = np.broadcast_to(std, (dim,))

    def _sampler(y_grid: Array, traj_idx: int) -> Array:
        del traj_idx
        return rng.normal(mean, std, size=y_grid.shape)

    return _sampler


def uniform_noise(
    *,
    bounds: float | Array,
    dim: int,
    rng: Rng = None,
) -> NoiseSampler:
    """Generate additive uniform observation noise."""
    rng = _require_rng(rng)
    bounds = np.broadcast_to(bounds, (dim, 2)).T

    def _sampler(y_grid: Array, traj_idx: int) -> Array:
        del traj_idx
        return rng.uniform(low=bounds[0], high=bounds[1], size=y_grid.shape)

    return _sampler


NOISE_MAP = {
    "gaussian": gaussian_noise,
    "uniform": uniform_noise,
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

    With input:

    .. math::
        \begin{align*}
        \dot{x} &= f(t, x, u) \\
        y &= g(t, x, u)
        \end{align*}

    Without input:

    .. math::
        \begin{align*}
        \dot{x} &= f(t, x) \\
        y &= g(t, x)
        \end{align*}

    Args:
        f (Callable[[float, Array, Array], Array]): Function defining the system dynamics.
            It should take time `t`, state `x`, and control input `u` as arguments and return the state derivative.
            Or just `f(t, x)` if the system is autonomous (no control input).
        g (Callable[[float, Array, Array], Array], optional): Function defining the observation
            model. It should take time `t`, state `x`, and control input `u` as arguments and return the observation.
            If not provided, it defaults to the identity function (`g(t, x, u) = x`).
            Or just `g(t, x)` if the system is autonomous (no control input).
        config (Union[str, Dict], optional): Path to a YAML configuration file or a dictionary
            containing the configuration for the sampler. The configuration should specify the dimensions
            of the states, inputs, and observations, as well as control and initial condition specifications.
        rng (Union[int, np.random.Generator, None], optional): Random number generator or seed for reproducibility.
            If an integer is provided, it is used to seed the default random number generator.
            If `None`, the default random generator is used.
        config_mod (Dict, optional): Additional configuration parameters to modify the loaded configuration.
            This should be a dictionary that updates or overrides the values in the loaded configuration.
    """

    def __init__(
        self,
        f: Callable[..., Array],
        g: Callable[..., Array] | None = None,
        config: ConfigLike = None,
        rng: int | np.random.Generator | None = None,
        config_mod: dict[str, Any] | None = None,
    ):
        self.f: Callable[..., Array] = f
        self.g: Callable[..., Array] = (lambda t, x, u=None: x) if g is None else g
        self.rng = np.random.default_rng(rng)
        self._noise_rng = _independent_rng(rng)

        if isinstance(config, dict):
            self.config = dict(config)
            if config_mod is not None:
                self.config.update(config_mod)
        else:
            self.config = load_config(str(config), config_mod)

        tmp = self.config.get("dims", None)
        if tmp is None:
            raise ValueError("Config must specify 'dims' (state/observation/input dimensions).")
        self.dims = [tmp["states"], tmp["inputs"], tmp["observations"], tmp.get("parameters", 0)]

        self._is_autonomous = self.dims[1] == 0
        if self._is_autonomous:
            self.sample: Callable[..., SampleWithControl | SampleAutonomous] = self._sample_auto
        else:
            self.sample = self._sample_ctrl

        self._n_skip = self.config.get("postprocess", {}).get("n_skip", 0)
        self._shift_t = self.config.get("postprocess", {}).get("shift_t", False)

        logger.info(
            f"TrajectorySampler initialized with dims: "
            f"states={self.dims[0]}, inputs={self.dims[1]}, "
            f"observations={self.dims[2]}, parameters={self.dims[3]}."
        )
        if self._is_autonomous:
            logger.info("Sampler is autonomous (no control inputs).")
        else:
            logger.info(f"Control config: {self.config.get('control', None)}")
        logger.info(f"Init. Cond. config: {self.config.get('x0', None)}")
        logger.info(f"Param. config: {self.config.get('p', None)}")
        logger.info(f"Noise config: {self.config.get('noise', None)}")
        logger.info(f"Solver config: {self.config.get('solver', None)}")
        logger.info(f"Postprocess config: {self.config.get('postprocess', None)}")

    def _create_control_sampler(
        self,
        t_grid: Array,
        traj_idx: int,
    ) -> tuple[Callable[[float | Array], Array], Array]:
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
            def u_call_external(t: float | Array, idx: int = traj_idx) -> Array:
                return np.asarray(u_spec(t, idx))

            u_grid = np.stack([u_call_external(t) for t in t_grid])
            return u_call_external, u_grid

        if isinstance(u_spec, Array):
            # Externally supplied array data
            U_vec = u_spec if u_spec.ndim == 2 else u_spec[traj_idx]
            assert t_grid.size == U_vec.shape[0], (
                f"t_grid size {t_grid.size} does not match U_vec size {U_vec.shape[0]}"
            )
            mode = "cubic"
            u_call_interp = _build_interpolant(t_grid, U_vec, mode)
            return u_call_interp, U_vec

        if isinstance(u_spec, dict):
            # Defined by a dictionary
            kind = u_spec["kind"].lower()
            if kind not in CTRL_MAP:
                raise KeyError(f"Unknown control kind '{kind}'. Available: {list(CTRL_MAP)}")
            params = copy.deepcopy(u_spec.get("params", {}))
            params.update({"dim": self.dims[1], "rng": self.rng})
            u_func = CTRL_MAP[kind](**params)

            def u_call_generated(t: float | Array, i: int = traj_idx) -> Array:
                return u_func(t, i)

            U_vec = u_call_generated(t_grid)
            return u_call_generated, U_vec.reshape(-1, self.dims[1])

        raise TypeError("Unrecognised u_spec type.")

    def _sample_xp(self, traj_num: int, pref: str = "x0", dims: int | None = None) -> Array | None:
        x0_spec = self.config.get(pref, None)
        if x0_spec is None:
            return None

        if isinstance(x0_spec, Array):
            # Externally supplied array data
            x0_arr = np.asarray(x0_spec)
            if x0_arr.ndim == 1:
                assert dims is not None and x0_arr.shape[0] == dims
                return np.stack([x0_arr.copy() for _ in range(traj_num)], axis=0)
            else:
                assert dims is not None and x0_arr.shape[1] == dims
                assert x0_arr.shape[0] >= traj_num
                return x0_arr[:traj_num]

        if isinstance(x0_spec, dict):
            # Defined by a dictionary
            kind = x0_spec["kind"].lower()
            if kind not in X0_MAP:
                raise KeyError(f"Unknown {pref} kind '{kind}'. Available: {list(X0_MAP)}")
            params = copy.deepcopy(x0_spec.get("params", {}))
            params.update({"dim": dims, "rng": self.rng})
            x0_func = X0_MAP[kind](**params)
            return np.asarray([x0_func(_i) for _i in range(traj_num)])

        raise TypeError("Unrecognised x0_spec type.")

    def _apply_observation_noise(self, y_grid: Array, traj_idx: int) -> Array:
        noise_spec = self.config.get("noise", None)
        if noise_spec is None:
            return y_grid

        if callable(noise_spec):
            return y_grid + np.asarray(noise_spec(y_grid, traj_idx))

        if not isinstance(noise_spec, dict) and not isinstance(noise_spec, (str, bytes)):
            noise_arr = np.asarray(noise_spec)
            if noise_arr.ndim == y_grid.ndim + 1:
                try:
                    return y_grid + np.broadcast_to(noise_arr[traj_idx], y_grid.shape)
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Noise array shape {noise_arr.shape} is incompatible with "
                        f"observation shape {y_grid.shape} for trajectory {traj_idx}."
                    ) from exc
            if noise_arr.ndim == y_grid.ndim and noise_arr.shape[1:] == y_grid.shape[1:]:
                if noise_arr.shape != y_grid.shape:
                    try:
                        return y_grid + np.broadcast_to(noise_arr[traj_idx], y_grid.shape)
                    except (IndexError, ValueError) as exc:
                        raise ValueError(
                            f"Noise array shape {noise_arr.shape} is incompatible with "
                            f"observation shape {y_grid.shape} for trajectory {traj_idx}."
                        ) from exc
            try:
                return y_grid + np.broadcast_to(noise_arr, y_grid.shape)
            except ValueError as exc:
                raise ValueError(
                    f"Noise array shape {noise_arr.shape} is incompatible with "
                    f"observation shape {y_grid.shape}."
                ) from exc

        if isinstance(noise_spec, dict):
            kind = noise_spec["kind"].lower()
            if kind not in NOISE_MAP:
                raise KeyError(f"Unknown noise kind '{kind}'. Available: {list(NOISE_MAP)}")
            params = copy.deepcopy(noise_spec.get("params", {}))
            params.update({"dim": self.dims[2], "rng": self._noise_rng})
            noise_func = NOISE_MAP[kind](**params)
            return y_grid + noise_func(y_grid, traj_idx)

        raise TypeError("Unrecognised noise_spec type.")

    def _sample_ctrl(
        self, t_samples: Array, batch: int = 1, save: str | None = None
    ) -> SampleWithControl:
        """
        Sample trajectories with control for a given time grid.

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
        Nt = tt.size - self._n_skip
        if Nt <= 0:
            raise ValueError(
                f"Time samples must be longer than n_skip.  Got {Nt} vs skip={self._n_skip}."
            )
        opts = self.config.get("solver", {})

        ts = np.zeros((batch, Nt))
        xs = np.zeros((batch, Nt, self.dims[0]))
        us = np.zeros((batch, Nt, self.dims[1]))
        ys = np.zeros((batch, Nt, self.dims[2]))
        x0s = self._sample_xp(batch, pref="x0", dims=self.dims[0])
        ps = self._sample_xp(batch, pref="p", dims=self.dims[3])
        if x0s is None:
            raise ValueError("Initial-condition sampler 'x0' is required for controlled sampling.")

        for i in range(batch):
            logger.info(f"Generating trajectory {i + 1}/{batch}...")

            fu, uu = self._create_control_sampler(tt, i)
            if ps is None:

                def rhs(t: float, x: Array) -> Array:
                    return np.asarray(self.f(t, x, fu(t)))
            else:

                def rhs(t: float, x: Array) -> Array:
                    return np.asarray(self.f(t, x, fu(t), ps[i]))

            sol = solve_ivp(rhs, (tt[0], tt[-1]), x0s[i], t_eval=tt, **opts)
            if not sol.success:
                raise RuntimeError(f"Integration failed on traj {i}: {sol.message}")
            xx = sol.y.T
            yy = self._apply_observation_noise(np.asarray(self.g(tt, xx, uu)), i)

            if self._shift_t:
                ts[i] = tt[self._n_skip :] - tt[self._n_skip]
            else:
                ts[i] = tt[self._n_skip :]
            xs[i], us[i], ys[i] = xx[self._n_skip :], uu[self._n_skip :], yy[self._n_skip :]

        if save is not None:
            assert isinstance(save, str), "Save path must be a string."
            os.makedirs(os.path.dirname(save), exist_ok=True)
            if ps is None:
                np.savez_compressed(save, t=ts, x=ys, u=us)
            else:
                np.savez_compressed(save, t=ts, x=ys, u=us, p=ps)

        if ps is None:
            return ts, xs, us, ys
        return ts, xs, us, ys, ps

    def _sample_auto(
        self, t_samples: Array, batch: int = 1, save: str | None = None
    ) -> SampleAutonomous:
        """
        Sample trajectories without control for a given time grid.

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
        Nt = tt.size - self._n_skip
        if Nt <= 0:
            raise ValueError(
                f"Time samples must be longer than n_skip.  Got {Nt} vs skip={self._n_skip}."
            )
        opts = self.config.get("solver", {})

        ts = np.zeros((batch, Nt))
        xs = np.zeros((batch, Nt, self.dims[0]))
        ys = np.zeros((batch, Nt, self.dims[2]))
        x0s = self._sample_xp(batch, pref="x0", dims=self.dims[0])
        ps = self._sample_xp(batch, pref="p", dims=self.dims[3])
        if x0s is None:
            raise ValueError("Initial-condition sampler 'x0' is required for autonomous sampling.")

        for i in range(batch):
            logger.info(f"Generating trajectory {i + 1}/{batch}...")

            if ps is None:
                sol = solve_ivp(cast(Any, self.f), (tt[0], tt[-1]), x0s[i], t_eval=tt, **opts)
            else:
                sol = solve_ivp(
                    cast(Any, self.f), (tt[0], tt[-1]), x0s[i], t_eval=tt, args=(ps[i],), **opts
                )
            if not sol.success:
                raise RuntimeError(f"Integration failed on traj {i}: {sol.message}")
            xx = sol.y.T
            yy = self._apply_observation_noise(np.asarray(self.g(tt, xx)), i)

            if self._shift_t:
                ts[i] = tt[self._n_skip :] - tt[self._n_skip]
            else:
                ts[i] = tt[self._n_skip :]
            xs[i], ys[i] = xx[self._n_skip :], yy[self._n_skip :]

        if save is not None:
            assert isinstance(save, str), "Save path must be a string."
            os.makedirs(os.path.dirname(save), exist_ok=True)
            if ps is None:
                np.savez_compressed(save, t=ts, x=ys)
            else:
                np.savez_compressed(save, t=ts, x=ys, p=ps)

        if ps is None:
            return ts, xs, ys
        return ts, xs, ys, ps
