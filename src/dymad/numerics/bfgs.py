"""Deterministic, closure-based BFGS optimizers with an owned strong-Wolfe search.

Each ``step`` performs one update. The closure must clear gradients, compute the
same deterministic scalar objective, and backpropagate it. Accepted loss and
gradient evaluations are reused across steps; call ``reset`` whenever the
objective changes. L-BFGS uses a fixed identity base, without secant rescaling.
"""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

_COUNTERS = (
    "func_evals",
    "n_iter",
    "line_search_evals",
    "nonfinite_trials",
    "curvature_updates",
    "curvature_skips",
    "hessian_resets",
)


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _options(group: dict) -> None:
    for name in ("lr", "tolerance_grad", "tolerance_change", "curvature_floor"):
        value = group[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be a finite real number")
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"invalid {name}: {value}")
    _positive_int("max_eval", group["max_eval"])
    if "history_size" in group:
        _positive_int("history_size", group["history_size"])


def _finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _spd(matrix: Tensor) -> bool:
    # CPU Cholesky does not support half/bfloat16; validation can be promoted.
    work = matrix if matrix.dtype in (torch.float32, torch.float64) else matrix.float()
    return bool(torch.linalg.cholesky_ex(work, check_errors=False).info.eq(0).all())


def _strong_wolfe(
    evaluate: Callable[[float], tuple[float, float] | None],
    value: float,
    slope: float,
    initial_step: float,
    displacement: float,
    tolerance: float,
) -> float:
    """Finite expansion and bisection zoom; ``evaluate`` owns the call budget.

    Nonfinite trials are upper boundaries, never substitute function values.
    An accepted point is always the last point evaluated. No interpolation or
    fallback can bypass Armijo or the strong curvature condition.
    """
    c1, c2 = 1e-4, 0.9

    def armijo(alpha: float, loss: float) -> bool:
        bound = value + c1 * alpha * slope
        return math.isfinite(bound) and loss <= bound

    def zoom(lo: float, hi: float, lo_value: float) -> float:
        while True:
            alpha = lo + 0.5 * (hi - lo)
            if (
                not math.isfinite(alpha)
                or alpha <= 0
                or alpha in (lo, hi)
                or abs(hi - lo) * displacement <= tolerance
            ):
                raise RuntimeError("strong-Wolfe line search bracket exhausted")
            trial = evaluate(alpha)
            if trial is None:
                hi = alpha
                continue
            loss, derivative = trial
            if not armijo(alpha, loss) or loss >= lo_value:
                hi = alpha
                continue
            if abs(derivative) <= -c2 * slope:
                return alpha
            if (derivative > 0) == (hi > lo):
                hi = lo
            lo, lo_value = alpha, loss

    previous, previous_value = 0.0, value
    alpha = initial_step
    while True:
        if not math.isfinite(alpha) or alpha <= 0:
            raise RuntimeError("strong-Wolfe line search step overflow")
        trial = evaluate(alpha)
        if trial is None:
            return zoom(previous, alpha, previous_value)
        loss, derivative = trial
        if not armijo(alpha, loss) or (previous > 0 and loss >= previous_value):
            return zoom(previous, alpha, previous_value)
        if abs(derivative) <= -c2 * slope:
            return alpha
        if derivative >= 0:
            return zoom(alpha, previous, loss)
        previous, previous_value = alpha, loss
        alpha *= 2.0


class BFGS(Optimizer):
    """Full inverse BFGS, one deterministic full-objective update per step.

    ``max_eval`` bounds every closure call in a step, including the initial
    evaluation when no cache exists. Failed searches raise and restore the
    accepted parameters and gradients. ``state_dict`` uses ordinary PyTorch
    parameter rebinding; load the matching model weights to resume curvature.
    External parameter changes automatically invalidate the coupled cache.
    """

    _kind = "bfgs"

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1.0,
        max_eval: int = 25,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
        curvature_floor: float = 1e-10,
    ):
        defaults = dict(
            lr=lr,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            curvature_floor=curvature_floor,
            algorithm=self._kind,
        )
        _options(defaults)
        super().__init__(params, defaults)
        self._parameters()
        _options(self.param_groups[0])

    def add_param_group(self, param_group: dict) -> None:
        if self.param_groups:
            raise ValueError("BFGS optimizers support exactly one parameter group")
        super().add_param_group(param_group)

    def _parameters(self) -> list[Tensor]:
        if len(self.param_groups) != 1:
            raise ValueError("BFGS optimizers support exactly one parameter group")
        params = self.param_groups[0]["params"]
        if not params or not sum(p.numel() for p in params):
            raise ValueError("BFGS requires nonempty parameters")
        reference = params[0]
        if len({id(p) for p in params}) != len(params):
            raise ValueError("BFGS parameters must not be duplicated")
        for param in params:
            if not param.is_floating_point() or not param.requires_grad:
                raise ValueError("BFGS requires real floating parameters with requires_grad=True")
            if param.dtype != reference.dtype or param.device != reference.device:
                raise ValueError("BFGS parameters must share dtype and device")
            if param.layout != torch.strided:
                raise ValueError("BFGS requires strided parameters")
        return params

    @property
    def status(self) -> str:
        return self.state.get(self.param_groups[0]["params"][0], {}).get("status", "running")

    def reset(self) -> None:
        """Discard curvature and cached evaluations after an objective change."""
        self.state.clear()

    @staticmethod
    def _assign(params: list[Tensor], vector: Tensor, *, gradient: bool = False) -> None:
        offset = 0
        with torch.no_grad():
            for param in params:
                chunk = vector[offset : offset + param.numel()].reshape(param.shape)
                if gradient:
                    param.grad = chunk.clone()
                else:
                    param.copy_(chunk)
                offset += param.numel()

    @staticmethod
    def _gradient(params: list[Tensor]) -> Tensor:
        if all(p.grad is None for p in params):
            raise RuntimeError("BFGS closure must backpropagate the scalar loss")
        if any(p.grad is not None and p.grad.layout != torch.strided for p in params):
            raise RuntimeError("BFGS does not support sparse gradients")
        return torch.cat(
            [
                (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in params
            ]
        ).clone()

    def _curvature(self, x: Tensor) -> dict:
        return {"inverse": torch.eye(x.numel(), dtype=x.dtype, device=x.device)}

    def _direction(self, state: dict, gradient: Tensor) -> Tensor:
        return -(state["inverse"] @ gradient)

    def _update(self, state: dict, s: Tensor, y: Tensor, sy: Tensor) -> dict:
        inverse = state["inverse"]
        hy = inverse @ y
        result = inverse + ((1 + (y @ hy) / sy) / sy) * torch.outer(s, s)
        result = result - (torch.outer(s, hy) + torch.outer(hy, s)) / sy
        result = 0.5 * (result + result.T)
        if not _finite(result) or bool((result.diag() <= 0).any()):
            return {}
        if (state["n_iter"] + 1) % 25 == 0 and not _spd(result):
            return {**self._curvature(s), "hessian_resets": state["hessian_resets"] + 1}
        return {"inverse": result}

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor:
        if closure is None:
            raise ValueError("BFGS requires a closure")
        params = self._parameters()
        group = self.param_groups[0]
        _options(group)
        x = torch.cat([p.detach().reshape(-1) for p in params]).clone()
        if not _finite(x):
            raise RuntimeError("BFGS parameters are nonfinite")
        state = self.state[params[0]]
        if state and not torch.equal(x, state["x"]):
            self.reset()
            state = self.state[params[0]]
        if not state:
            state.update(
                {key: 0 for key in _COUNTERS},
                version=1,
                algorithm=self._kind,
                layout=[list(p.shape) for p in params],
                status="running",
            )
        incoming_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
        calls = 0

        def evaluate(point: Tensor, *, trial: bool) -> tuple[Tensor, Tensor] | None:
            nonlocal calls
            if calls >= group["max_eval"]:
                raise RuntimeError("strong-Wolfe line search exceeded max_eval")
            self._assign(params, point)
            self.zero_grad(set_to_none=True)
            calls += 1
            state["func_evals"] += 1
            state["line_search_evals"] += int(trial)
            try:
                with torch.enable_grad():
                    loss = closure()
                if (
                    not isinstance(loss, Tensor)
                    or loss.numel() != 1
                    or not loss.is_floating_point()
                ):
                    raise RuntimeError("BFGS closure must return a real scalar Tensor")
                loss = loss.detach().reshape(()).clone()
                gradient = self._gradient(params)
                if _finite(loss) and _finite(gradient):
                    return loss, gradient
            except FloatingPointError:
                if not trial:
                    raise
            if trial:
                state["nonfinite_trials"] += 1
            return None

        try:
            if "x" not in state:
                evaluated = evaluate(x, trial=False)
                if evaluated is None:
                    raise RuntimeError("BFGS initial loss or gradient is nonfinite")
                loss, gradient = evaluated
                state.update(x=x, loss=loss, grad=gradient, **self._curvature(x))
            loss, gradient = state["loss"], state["grad"]
            self._assign(params, gradient, gradient=True)
            if float(gradient.abs().max()) <= group["tolerance_grad"]:
                state["status"] = "converged"
                return loss.clone()
            state["status"] = "running"
            direction = self._direction(state, gradient)
            slope = float(gradient @ direction)
            reset_direction = not _finite(direction) or not math.isfinite(slope) or slope >= 0
            working = state
            if reset_direction:
                direction = -gradient
                slope = float(gradient @ direction)
                working = {**state, **self._curvature(x)}
            if not math.isfinite(slope) or slope >= 0:
                raise RuntimeError("BFGS could not obtain a finite descent direction")
            accepted: tuple[Tensor, Tensor, Tensor] | None = None
            attempts = 0
            remaining = group["max_eval"] - calls

            def trial(alpha: float) -> tuple[float, float] | None:
                nonlocal accepted, attempts
                if attempts >= remaining:
                    raise RuntimeError("strong-Wolfe line search exceeded max_eval")
                attempts += 1
                point = x + alpha * direction
                if not _finite(point):
                    # Invalid parameter proposals also consume the search budget,
                    # but do not invoke the closure or count as function calls.
                    state["nonfinite_trials"] += 1
                    return None
                result = evaluate(point, trial=True)
                if result is None:
                    return None
                trial_loss, trial_gradient = result
                derivative = float(trial_gradient @ direction)
                if not math.isfinite(derivative):
                    state["nonfinite_trials"] += 1
                    return None
                accepted = point, trial_loss, trial_gradient
                return float(trial_loss), derivative

            initial = group["lr"]
            if state["n_iter"] == 0:
                initial *= min(1.0, 1.0 / float(gradient.abs().sum()))
            _strong_wolfe(
                trial,
                float(loss),
                slope,
                initial,
                float(direction.abs().max()),
                group["tolerance_change"],
            )
            assert accepted is not None
            new_x, new_loss, new_gradient = accepted
            s, y = new_x - x, new_gradient - gradient
            sy = s @ y
            threshold = (
                group["curvature_floor"] * torch.linalg.vector_norm(s) * torch.linalg.vector_norm(y)
            )
            curvature = self._curvature(x) if reset_direction else {}
            if reset_direction:
                curvature["hessian_resets"] = state["hessian_resets"] + 1
                working = {**working, **curvature}
            update = {}
            if _finite(sy) and _finite(threshold) and float(sy) > float(threshold):
                update = self._update(working, s, y, sy)
            state.update(curvature, **update)
            state["curvature_updates" if update else "curvature_skips"] += 1
            state.update(x=new_x, loss=new_loss, grad=new_gradient)
            state["n_iter"] += 1
            state["status"] = (
                "converged"
                if float(new_gradient.abs().max()) <= group["tolerance_grad"]
                else "running"
            )
            self._assign(params, new_gradient, gradient=True)
            return new_loss.clone()
        except BaseException:
            self._assign(params, state.get("x", x))
            if "grad" in state:
                self._assign(params, state["grad"], gradient=True)
            else:
                for param, grad in zip(params, incoming_grads, strict=True):
                    param.grad = grad
                # A failed initial closure leaves no reusable numerical state.
                self.reset()
            raise

    def _validate_curvature(self, state: dict, n: int) -> None:
        matrix = state.get("inverse")
        if (
            not isinstance(matrix, Tensor)
            or matrix.shape != (n, n)
            or matrix.layout != torch.strided
            or matrix.dtype != state["x"].dtype
            or not _finite(matrix)
        ):
            raise ValueError("invalid BFGS inverse matrix")
        if not torch.allclose(matrix, matrix.T) or not _spd(matrix):
            raise ValueError("BFGS inverse matrix must be symmetric positive definite")

    def load_state_dict(self, state_dict: dict) -> None:
        """Validate cached state before standard PyTorch parameter rebinding."""
        params = self._parameters()
        groups = state_dict.get("param_groups", [])
        if len(groups) != 1 or groups[0].get("algorithm") != self._kind:
            raise ValueError("incompatible BFGS optimizer state")
        group = groups[0]
        _options(group)
        if len(group["params"]) != len(params):
            raise ValueError("incompatible BFGS parameter count")
        states = state_dict.get("state", {})
        if states:
            if set(states) != {group["params"][0]}:
                raise ValueError("BFGS state must belong to the first parameter")
            state = states[group["params"][0]]
            n = sum(p.numel() for p in params)
            if (
                state.get("version") != 1
                or state.get("algorithm") != self._kind
                or state.get("layout") != [list(p.shape) for p in params]
                or state.get("status") not in ("running", "converged")
            ):
                raise ValueError("incompatible BFGS state layout")
            for key, shape in (("x", (n,)), ("grad", (n,)), ("loss", ())):
                tensor = state.get(key)
                if (
                    not isinstance(tensor, Tensor)
                    or tensor.shape != shape
                    or tensor.layout != torch.strided
                    or not tensor.is_floating_point()
                    or not _finite(tensor)
                ):
                    raise ValueError(f"invalid BFGS cached {key}")
                if key != "loss" and tensor.dtype != params[0].dtype:
                    raise ValueError("BFGS cached state dtype must match model parameters")
            for key in _COUNTERS:
                value = state.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid BFGS counter {key}")
            self._validate_curvature(state, n)
            if self._kind == "lbfgs" and len(state["s_history"]) > group["history_size"]:
                raise ValueError("L-BFGS history exceeds history_size")
        super().load_state_dict(state_dict)
        if states:
            rebound = self.state[params[0]]
            # Optimizer's generic recursive casting treats strings as iterables
            # in some PyTorch versions. These fields are validated metadata.
            rebound["algorithm"] = state["algorithm"]
            rebound["status"] = state["status"]
            # A closure may accumulate a higher-precision scalar loss. Preserve
            # that precision instead of casting it to the parameter dtype.
            rebound["loss"] = state["loss"].to(device=params[0].device).clone()


class LBFGS(BFGS):
    """Limited-memory BFGS with the same search/cache and a fixed identity base."""

    _kind = "lbfgs"

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1.0,
        max_eval: int = 25,
        tolerance_grad: float = 1e-9,
        tolerance_change: float = 1e-12,
        curvature_floor: float = 1e-10,
        history_size: int = 100,
    ):
        _positive_int("history_size", history_size)
        super().__init__(params, lr, max_eval, tolerance_grad, tolerance_change, curvature_floor)
        self.defaults["history_size"] = history_size
        self.param_groups[0].setdefault("history_size", history_size)
        _options(self.param_groups[0])

    def _curvature(self, x: Tensor) -> dict:
        return {"s_history": [], "y_history": [], "rho_history": []}

    def _direction(self, state: dict, gradient: Tensor) -> Tensor:
        q = gradient.clone()
        pairs = list(zip(state["s_history"], state["y_history"], state["rho_history"], strict=True))
        coefficients = []
        for s, y, rho in reversed(pairs):
            alpha = rho * (s @ q)
            coefficients.append(alpha)
            q = q - alpha * y
        for (s, y, rho), alpha in zip(pairs, reversed(coefficients), strict=True):
            q = q + s * (alpha - rho * (y @ q))
        return -q

    def _update(self, state: dict, s: Tensor, y: Tensor, sy: Tensor) -> dict:
        rho = sy.reciprocal()
        if not _finite(rho):
            return {}
        size = self.param_groups[0]["history_size"]
        return {
            "s_history": (state["s_history"] + [s.clone()])[-size:],
            "y_history": (state["y_history"] + [y.clone()])[-size:],
            "rho_history": (state["rho_history"] + [rho.clone()])[-size:],
        }

    def _validate_curvature(self, state: dict, n: int) -> None:
        histories: list[list] = []
        for name in ("s_history", "y_history", "rho_history"):
            history = state.get(name)
            if not isinstance(history, list):
                raise ValueError("invalid L-BFGS history")
            histories.append(history)
        if len({len(history) for history in histories}) != 1:
            raise ValueError("incompatible L-BFGS history lengths")
        for s, y, rho in zip(*histories, strict=True):
            if any(
                not isinstance(t, Tensor)
                or t.layout != torch.strided
                or t.dtype != state["x"].dtype
                or not _finite(t)
                for t in (s, y, rho)
            ):
                raise ValueError("invalid L-BFGS history tensors")
            if s.shape != (n,) or y.shape != (n,) or rho.shape != ():
                raise ValueError("incompatible L-BFGS history shapes")
            sy = s @ y
            if float(sy) <= 0 or float(rho) <= 0 or not torch.allclose(rho * sy, rho.new_tensor(1)):
                raise ValueError("invalid L-BFGS secant curvature")
