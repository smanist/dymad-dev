"""Exact numerical and transactional contracts for local quasi-Newton methods."""

import copy
import pickle

import pytest
import torch

from dymad.numerics import BFGS, LBFGS


def _problem(optimizer_type, **options):
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64))
    matrix = torch.tensor([[8.0, 1.0, 0.2], [1.0, 3.0, 0.0], [0.2, 0.0, 1.0]], dtype=torch.float64)
    optimizer = optimizer_type([parameter], **options)
    evaluations = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * parameter @ matrix @ parameter
        loss.backward()
        evaluations.append(
            (parameter.detach().clone(), loss.detach().clone(), parameter.grad.clone())
        )
        return loss

    return parameter, matrix, optimizer, closure, evaluations


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_quadratic_steps_satisfy_wolfe_and_reuse_accepted_evaluations(optimizer_type):
    parameter, matrix, optimizer, closure, evaluations = _problem(optimizer_type)
    assert not optimizer.state
    for _ in range(30):
        x0 = parameter.detach().clone()
        f0 = 0.5 * x0 @ matrix @ x0
        g0 = matrix @ x0
        before = len(evaluations)
        # A newly created closure object must not invalidate the accepted cache.
        loss = optimizer.step(lambda: closure())
        displacement = parameter.detach() - x0
        if bool(displacement.ne(0).any()):
            slope = g0 @ displacement
            assert loss <= f0 + 1e-4 * slope + 1e-14
            assert abs(parameter.grad @ displacement) <= 0.9 * abs(slope) + 1e-14
            assert torch.equal(evaluations[-1][0], parameter)
            assert torch.equal(evaluations[-1][2], parameter.grad)
            if before:
                assert not torch.equal(evaluations[before][0], x0)
        if optimizer.status == "converged":
            break
    assert optimizer.status == "converged"
    assert float(loss) < 1e-17
    state = optimizer.state[parameter]
    assert state["func_evals"] == len(evaluations)
    assert state["func_evals"] == state["line_search_evals"] + 1
    before = len(evaluations)
    optimizer.step(closure)
    assert len(evaluations) == before


def test_full_and_untruncated_limited_follow_same_iterates():
    full = _problem(BFGS, tolerance_grad=1e-12)
    limited = _problem(LBFGS, tolerance_grad=1e-12, history_size=30)
    for _ in range(8):
        full[2].step(full[3])
        limited[2].step(limited[3])
        torch.testing.assert_close(full[0], limited[0], rtol=1e-8, atol=1e-12)
        torch.testing.assert_close(full[0].grad, limited[0].grad, rtol=1e-8, atol=1e-12)


def test_limited_two_loop_uses_identity_base_after_eviction():
    parameter, _, optimizer, closure, _ = _problem(LBFGS, history_size=2)
    for _ in range(4):
        optimizer.step(closure)
    state = optimizer.state[parameter]
    assert len(state["s_history"]) == 2
    identity = torch.eye(parameter.numel(), dtype=parameter.dtype)
    inverse = identity.clone()
    for s, y, rho in zip(state["s_history"], state["y_history"], state["rho_history"], strict=True):
        left = identity - rho * torch.outer(s, y)
        inverse = left @ inverse @ left.T + rho * torch.outer(s, s)
    torch.testing.assert_close(
        optimizer._direction(state, state["grad"]), -(inverse @ state["grad"])
    )


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_nonfinite_trials_are_bracketed_without_invalid_commit(optimizer_type):
    parameter = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    optimizer = optimizer_type([parameter])
    visited = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = -torch.log(parameter).sum() + 10 * parameter.sum()
        loss.backward()
        visited.append(parameter.detach().clone())
        return loss

    loss = optimizer.step(closure)
    assert torch.isfinite(loss)
    assert optimizer.state[parameter]["nonfinite_trials"] == 2
    assert any(float(point) < 0 for point in visited)
    assert torch.equal(visited[-1], parameter)
    assert float(parameter.detach()) > 0
    torch.testing.assert_close(parameter.grad, 10 - 1 / parameter)


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_evaluation_cap_rolls_back_and_keeps_reusable_base(optimizer_type):
    parameter, _, optimizer, closure, evaluations = _problem(optimizer_type, max_eval=1)
    start = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="max_eval"):
        optimizer.step(closure)
    assert torch.equal(parameter, start)
    assert torch.equal(parameter.grad, evaluations[0][2])
    assert optimizer.status == "running"
    assert optimizer.state[parameter]["n_iter"] == 0
    assert optimizer.state[parameter]["func_evals"] == 1
    optimizer.param_groups[0]["max_eval"] = 25
    optimizer.step(closure)
    assert not torch.equal(evaluations[1][0], start)
    assert optimizer.state[parameter]["func_evals"] == len(evaluations)


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_finite_loss_with_nonfinite_gradient_is_rejected(optimizer_type):
    parameter = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    optimizer = optimizer_type([parameter])
    losses = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - 0.1).square().sum()
        loss.backward()
        if float(parameter.detach()) < 0:
            parameter.grad.fill_(float("nan"))
        losses.append(loss.detach())
        return loss

    optimizer.step(closure)
    assert all(torch.isfinite(loss) for loss in losses)
    assert optimizer.state[parameter]["nonfinite_trials"] == 1
    assert torch.isfinite(parameter.grad).all()
    assert float(parameter.detach()) > 0


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_impossible_wolfe_never_accepts_a_merely_lower_loss(optimizer_type):
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = optimizer_type([parameter], max_eval=7)
    calls = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.sum()
        loss.backward()
        calls.append(float(loss.detach()))
        return loss

    with pytest.raises(RuntimeError, match="max_eval"):
        optimizer.step(closure)
    assert len(calls) == 7
    assert calls[-1] < calls[0]
    assert float(parameter.detach()) == 1
    assert float(parameter.grad) == 1
    assert optimizer.status == "running"
    assert optimizer.state[parameter]["n_iter"] == 0


def test_failed_initial_evaluation_restores_incoming_gradients():
    parameter = torch.nn.Parameter(torch.tensor([-1.0], dtype=torch.float64))
    parameter.grad = torch.tensor([7.0], dtype=torch.float64)
    optimizer = BFGS([parameter])

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.log(parameter).sum()
        loss.backward()
        return loss

    with pytest.raises(RuntimeError, match="initial loss or gradient"):
        optimizer.step(closure)
    assert float(parameter.detach()) == -1
    assert float(parameter.grad) == 7
    assert not optimizer.state


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_closure_exception_restores_parameters_gradients_and_curvature(optimizer_type):
    parameter, _, optimizer, closure, _ = _problem(optimizer_type)
    optimizer.step(closure)
    saved = copy.deepcopy(optimizer.state[parameter])
    start, grad = parameter.detach().clone(), parameter.grad.clone()

    def broken_closure():
        parameter.grad = torch.full_like(parameter, 999)
        raise ValueError("model error")

    with pytest.raises(ValueError, match="model error"):
        optimizer.step(broken_closure)
    assert torch.equal(parameter, start)
    assert torch.equal(parameter.grad, grad)
    state = optimizer.state[parameter]
    assert state["n_iter"] == saved["n_iter"]
    if optimizer_type is BFGS:
        assert torch.equal(state["inverse"], saved["inverse"])
    else:
        for actual, expected in zip(state["s_history"], saved["s_history"], strict=True):
            assert torch.equal(actual, expected)
    assert state["func_evals"] == saved["func_evals"] + 1


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_state_dict_rebind_and_whole_optimizer_pickle_resume(optimizer_type):
    parameter, matrix, optimizer, closure, _ = _problem(optimizer_type)
    for _ in range(3):
        optimizer.step(closure)
    saved = copy.deepcopy(optimizer.state_dict())
    rebound = torch.nn.Parameter(parameter.detach().clone())
    restored = optimizer_type([rebound])
    restored.load_state_dict(saved)
    assert restored.status == optimizer.status
    # Metadata must survive repeated checkpoint round trips, not just a step.
    restored.load_state_dict(copy.deepcopy(restored.state_dict()))
    assert restored.state[rebound]["algorithm"] == optimizer.state[parameter]["algorithm"]
    pickled = pickle.loads(pickle.dumps(optimizer))
    unpickled_parameter = pickled.param_groups[0]["params"][0]
    expected_loss = optimizer.step(closure)
    for candidate, p in ((restored, rebound), (pickled, unpickled_parameter)):
        calls = []

        def candidate_closure():
            candidate.zero_grad(set_to_none=True)
            loss = 0.5 * p @ matrix @ p
            loss.backward()
            calls.append(p.detach().clone())
            return loss

        actual_loss = candidate.step(candidate_closure)
        assert not torch.equal(calls[0], saved["state"][0]["x"])
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(p, parameter, rtol=0, atol=0)
        torch.testing.assert_close(p.grad, parameter.grad, rtol=0, atol=0)
        assert candidate.state[p]["func_evals"] == optimizer.state[parameter]["func_evals"]


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_reset_and_external_parameter_changes_clear_coupled_cache(optimizer_type):
    parameter, _, optimizer, closure, evaluations = _problem(optimizer_type)
    optimizer.step(closure)
    optimizer.reset()
    assert not optimizer.state
    assert optimizer.status == "running"
    before = len(evaluations)
    optimizer.step(closure)
    assert len(evaluations) > before + 1
    with torch.no_grad():
        parameter.add_(0.5)
    changed = parameter.detach().clone()
    before = len(evaluations)
    optimizer.step(closure)
    assert torch.equal(evaluations[before][0], changed)
    assert optimizer.state[parameter]["n_iter"] == 1


def test_multiple_parameter_shapes_and_unused_parameters():
    used = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64).T)
    unused = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    optimizer = BFGS([used, unused])

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = used.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    assert torch.equal(unused, torch.ones_like(unused))
    assert torch.equal(unused.grad, torch.zeros_like(unused))
    assert used.grad.shape == used.shape


@pytest.mark.parametrize("optimizer_type", [BFGS, LBFGS])
def test_corrupt_or_wrong_algorithm_checkpoint_rejected(optimizer_type):
    parameter, _, optimizer, closure, _ = _problem(optimizer_type)
    optimizer.step(closure)
    saved = copy.deepcopy(optimizer.state_dict())
    state = saved["state"][0]
    state["grad"][0] = float("nan")
    with pytest.raises(ValueError, match="cached grad"):
        optimizer.load_state_dict(saved)
    saved = copy.deepcopy(optimizer.state_dict())
    saved["state"][0]["layout"] = [[1, 3]]
    with pytest.raises(ValueError, match="layout"):
        optimizer.load_state_dict(saved)
    saved = copy.deepcopy(optimizer.state_dict())
    if optimizer_type is BFGS:
        saved["state"][0]["inverse"][0, 0] = -1
        message = "positive definite"
    else:
        saved["state"][0]["rho_history"][0].neg_()
        message = "curvature"
    with pytest.raises(ValueError, match=message):
        optimizer.load_state_dict(saved)
    other = LBFGS if optimizer_type is BFGS else BFGS
    with pytest.raises(ValueError, match="incompatible"):
        other([parameter]).load_state_dict(optimizer.state_dict())
    different_dtype = torch.nn.Parameter(parameter.detach().float())
    with pytest.raises(ValueError, match="dtype"):
        optimizer_type([different_dtype]).load_state_dict(optimizer.state_dict())


@pytest.mark.parametrize(
    "options",
    [
        {"lr": 0},
        {"lr": float("inf")},
        {"max_eval": 0},
        {"max_eval": True},
        {"max_eval": 1.5},
        {"tolerance_grad": -1},
        {"tolerance_change": float("nan")},
        {"curvature_floor": -1},
    ],
)
def test_option_guards(options):
    with pytest.raises(ValueError):
        BFGS([torch.nn.Parameter(torch.ones(1))], **options)


def test_parameter_group_and_history_guards():
    p = torch.nn.Parameter(torch.ones(1))
    q = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="one parameter group"):
        BFGS([{"params": [p]}, {"params": [q]}])
    with pytest.raises(ValueError, match="dtype"):
        BFGS([p, q])
    with pytest.raises(ValueError, match="requires_grad"):
        BFGS([torch.ones(1)])
    with pytest.raises(ValueError, match="real"):
        BFGS([torch.nn.Parameter(torch.ones(1, dtype=torch.complex64))])
    with pytest.raises(ValueError, match="history_size"):
        LBFGS([p], history_size=0)
    optimizer = BFGS([p])
    with pytest.raises(ValueError, match="one parameter group"):
        optimizer.add_param_group({"params": [q]})
