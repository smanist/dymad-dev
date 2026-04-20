import numpy as np

from dymad.numerics import disc2cont
from dymad.sako.adapter import SpectralAnalysisAdapter, SpectralEigensystem
from dymad.sako.snapshot import build_spectral_snapshot


def _build_snapshot():
    return build_spectral_snapshot(
        model_class="DKBF",
        checkpoint_path="checkpoints/model.pt",
        encoded_p0=np.arange(12, dtype=float).reshape(6, 2),
        encoded_p1=np.arange(12, dtype=float).reshape(6, 2) + 1.0,
        weights=(np.eye(2),),
        input_dim=2,
        obs_dim=2,
        metadata={"processor_mode": "full"},
    )


def _build_eigensystem():
    return SpectralEigensystem(
        discrete_eigs=np.array([0.9 + 0.2j, 0.8 - 0.1j]),
        left_eigvecs=np.array([[1.0, 2.0], [3.0, 4.0]]),
        right_eigvecs=np.array([[0.5, 0.0], [0.0, 0.25]]),
        projector=np.eye(2),
        dt=0.2,
    )


def test_spectral_adapter_initializes_kernels_from_snapshot(monkeypatch):
    captured = {}

    class DummySAKO:
        def __init__(self, p0, p1, _, reps, etol):
            captured["sako"] = {
                "p0": np.array(p0, copy=True),
                "p1": np.array(p1, copy=True),
                "reps": reps,
                "etol": etol,
            }

        def estimate_measure(self, *_args, **_kwargs):
            return np.array([1.0])

    class DummyRALowRank:
        def __init__(self, vr, diag_wc, vl, dt):
            captured["rals"] = {
                "vr": np.array(vr, copy=True),
                "diag_wc": np.array(diag_wc, copy=True),
                "vl": np.array(vl, copy=True),
                "dt": dt,
            }

        def __call__(self, z, return_vec, mode):
            return z, return_vec, mode

    monkeypatch.setattr("dymad.sako.adapter.SAKO", DummySAKO)
    monkeypatch.setattr("dymad.sako.adapter.RALowRank", DummyRALowRank)

    snapshot = _build_snapshot()
    eigensystem = _build_eigensystem()
    adapter = SpectralAnalysisAdapter(snapshot=snapshot, eigensystem=eigensystem)

    assert isinstance(adapter.sako, DummySAKO)
    assert isinstance(adapter.rals, DummyRALowRank)
    assert np.array_equal(captured["sako"]["p0"], snapshot.encoded_p0)
    assert np.array_equal(captured["sako"]["p1"], snapshot.encoded_p1)
    expected_wc = disc2cont(eigensystem.discrete_eigs, eigensystem.dt).conj()
    assert np.allclose(np.diag(captured["rals"]["diag_wc"]), expected_wc)
    assert captured["rals"]["dt"] == 0.2


def test_spectral_adapter_delegates_measure_and_jacobian_calls(monkeypatch):
    captured = {}

    class DummySAKO:
        def __init__(self, *_args, **_kwargs):
            pass

        def estimate_measure(self, gobs, order, eps, thetas):
            captured["measure"] = {
                "gobs": np.array(gobs, copy=True),
                "order": order,
                "eps": eps,
                "thetas": thetas,
            }
            return np.array([3.0, 4.0])

    class DummyRALowRank:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, z, return_vec, mode):
            return z, return_vec, mode

    class Runtime:
        def apply_obs(self, fobs):
            return np.asarray(fobs) + 5.0

        def get_forward_modes(self, ref, rng=None, **kwargs):
            captured["forward_ref"] = np.array(ref, copy=True)
            return np.array([[2.0, 0.0], [0.0, 1.0]])

        def get_backward_modes(self, ref, rng=None, **kwargs):
            captured["backward_ref"] = np.array(ref, copy=True)
            return np.array([[1.0, 1.0], [0.0, 2.0]])

    monkeypatch.setattr("dymad.sako.adapter.SAKO", DummySAKO)
    monkeypatch.setattr("dymad.sako.adapter.RALowRank", DummyRALowRank)

    snapshot = _build_snapshot()
    eigensystem = _build_eigensystem()
    adapter = SpectralAnalysisAdapter(
        snapshot=snapshot,
        eigensystem=eigensystem,
        runtime=Runtime(),
    )

    measure = adapter.estimate_measure(np.array([[1.0, 2.0]]), order=9, eps=1e-3, thetas=17)
    eigfunc_jac = adapter.eval_eigfunc_jac()
    eigmode_jac = adapter.eval_eigmode_jac()

    assert np.array_equal(measure, np.array([3.0, 4.0]))
    assert np.array_equal(captured["measure"]["gobs"], np.array([6.0, 7.0]))
    assert np.array_equal(captured["forward_ref"], np.zeros((1, snapshot.input_dim)))
    assert np.array_equal(captured["backward_ref"], np.zeros((1, snapshot.obs_dim)))
    assert np.array_equal(
        eigfunc_jac, eigensystem.left_eigvecs.T.dot(np.array([[2.0, 0.0], [0.0, 1.0]]))
    )
    assert np.array_equal(
        eigmode_jac, eigensystem.right_eigvecs.T.dot(np.array([[1.0, 1.0], [0.0, 2.0]]))
    )


def test_spectral_adapter_delegates_pseudospectrum_estimation(monkeypatch):
    captured = {}

    class DummySAKO:
        def __init__(self, *_args, **_kwargs):
            pass

        def _ps_point(self, z, return_vec):
            if return_vec:
                return 2.0, np.array([1.0, -1.0])
            return 2.0

    class DummyRALowRank:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, z, return_vec, mode):
            captured["rals_call"] = {"z": z, "return_vec": return_vec, "mode": mode}
            return np.array([z.real + z.imag])

    def fake_estimate_pseudospectrum(grid, estimator, return_vec=False, **kwargs):
        captured["ps"] = {
            "grid": np.array(grid, copy=True),
            "return_vec": return_vec,
            "kwargs": dict(kwargs),
        }
        probe = estimator(0.3 + 0.7j, return_vec, kwargs["mode"], kwargs["method"])
        return {"probe": probe}

    monkeypatch.setattr("dymad.sako.adapter.SAKO", DummySAKO)
    monkeypatch.setattr("dymad.sako.adapter.RALowRank", DummyRALowRank)
    monkeypatch.setattr("dymad.sako.adapter.estimate_pseudospectrum", fake_estimate_pseudospectrum)

    adapter = SpectralAnalysisAdapter(snapshot=_build_snapshot(), eigensystem=_build_eigensystem())
    grid, result = adapter.estimate_ps(
        grid=np.array([0.1 + 0.2j, 0.3 + 0.4j]),
        return_vec=False,
        mode="disc",
        method="standard",
    )

    assert np.array_equal(grid, np.array([0.1 + 0.2j, 0.3 + 0.4j]))
    assert captured["ps"]["kwargs"] == {"mode": "disc", "method": "standard"}
    assert captured["rals_call"]["mode"] == "disc"
    assert result == {"probe": np.array([1.0])}
