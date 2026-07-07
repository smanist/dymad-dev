import os
from pathlib import Path

import pytest
import torch

from dymad.modules import KernelScDM, KernelScDMHeat

os.environ.setdefault("KEOPS_CACHE_FOLDER", str(Path("/tmp") / "dymad_keops_cache"))


def test_scdm_forward_preserves_symmetric_normalized_kernel():
    X = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [1.0, 1.0]], dtype=torch.float64)
    kernel = KernelScDM(in_dim=2, eps_init=0.2, alpha_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(X)

    W = torch.exp(-(torch.cdist(X, X) ** 2) / (4.0 * kernel.eps))
    q = W.sum(dim=-1)
    D = q ** (-kernel.alpha)
    expected = D[:, None] * W * D[None, :]
    Dinv1 = expected.sum(dim=-1) ** (-0.5)
    expected = Dinv1[:, None] * expected * Dinv1[None, :]

    actual = kernel(X, None)
    assert actual.shape == (4, 4)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, actual.T)


def test_scdm_periodic_metric_wraps_unit_period_axes():
    X = torch.tensor([[0.01]], dtype=torch.float64)
    Z = torch.tensor([[0.99]], dtype=torch.float64)
    eps = torch.tensor(0.1, dtype=torch.float64)

    euclidean = KernelScDM(
        in_dim=1, eps_init=0.1, metric="euclidean", dtype=torch.float64
    )._raw_kernel(X, Z, eps=eps)
    periodic = KernelScDM(
        in_dim=1, eps_init=0.1, metric="periodic", dtype=torch.float64
    )._raw_kernel(X, Z, eps=eps)
    periodic_axis_subset = KernelScDM(
        in_dim=2,
        eps_init=0.1,
        metric="periodic",
        periodic_axes=(0,),
        dtype=torch.float64,
    )._raw_kernel(
        torch.tensor([[0.01, 0.01]], dtype=torch.float64),
        torch.tensor([[0.99, 0.99]], dtype=torch.float64),
        eps=eps,
    )

    assert torch.allclose(
        euclidean, torch.exp(torch.tensor([[-0.9604 / 0.4]], dtype=torch.float64))
    )
    assert torch.allclose(periodic, torch.exp(torch.tensor([[-0.0004 / 0.4]], dtype=torch.float64)))
    assert torch.allclose(
        periodic_axis_subset, torch.exp(torch.tensor([[-0.9608 / 0.4]], dtype=torch.float64))
    )


def test_scdm_periodic_axes_must_be_integers():
    with pytest.raises(TypeError, match="integers"):
        KernelScDM(
            in_dim=2,
            eps_init=0.1,
            metric="periodic",
            periodic_axes=(0.9,),
            dtype=torch.float64,
        )


def test_scdm_density_heat_kernel_direct_reference_has_unit_weight_mass():
    Xref = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.02], [0.37], [0.89]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.01,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
        density_bandwidth_factor=1.5,
    )
    kernel.set_reference_data(Xref)

    q_ref = kernel._raw_kernel(Xref, Xref, eps=kernel.density_eps).sum(dim=-1)
    weights = q_ref.reciprocal() / q_ref.reciprocal().sum()
    heat = kernel.heat_kernel(Xref, sources, mode="density", location_weights=weights)
    mass = (heat * weights[:, None]).sum(dim=0)

    assert heat.shape == (24, 3)
    assert weights.shape == (24,)
    assert torch.all(torch.isfinite(heat))
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(mass, torch.ones_like(mass))


def test_scdm_density_heat_kernel_is_finite_for_nonuniform_reference_points():
    Xref = torch.tensor(
        [[0.00], [0.01], [0.015], [0.20], [0.55], [0.73], [0.95]], dtype=torch.float64
    )
    sources = torch.tensor([[0.02], [0.40], [0.98]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.02,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    q_ref = kernel._raw_kernel(Xref, Xref, eps=kernel.density_eps).sum(dim=-1)
    weights = q_ref.reciprocal() / q_ref.reciprocal().sum()
    heat = kernel.heat_kernel(Xref, sources, mode="density", alpha=1.0, location_weights=weights)
    mass = (heat * weights[:, None]).sum(dim=0)

    assert torch.all(torch.isfinite(heat))
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(mass, torch.ones_like(mass))


def test_scdm_density_heat_kernel_direct_reference_matches_manual_density_formula():
    Xref = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.03], [0.41], [0.88]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.02,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    heat = kernel.heat_kernel(
        Xref,
        sources,
        mode="density",
        steps=1,
        alpha=1.0,
        mass_normalization="none",
    )
    q_sources = kernel._raw_kernel(sources, Xref, eps=kernel.density_eps).sum(dim=-1)
    q_ref = kernel._raw_kernel(Xref, Xref, eps=kernel.density_eps).sum(dim=-1)
    weights = q_ref.reciprocal() / q_ref.reciprocal().sum()
    block = kernel._raw_kernel(sources, Xref) / (q_sources[:, None] * q_ref[None, :])
    markov = block / block.sum(dim=-1)[:, None]
    expected = markov / weights[None, :]

    assert heat.shape == (16, 3)
    assert torch.allclose(heat, expected.T)


def test_scdm_density_heat_kernel_preserves_mass_on_reference_targets():
    Xref = torch.tensor(
        [[0.00], [0.01], [0.02], [0.21], [0.50], [0.74], [0.96]], dtype=torch.float64
    )
    sources = torch.tensor([[0.015], [0.42], [0.90]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.03,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    q_ref = kernel._raw_kernel(Xref, Xref, eps=kernel.density_eps).sum(dim=-1)
    weights = q_ref.reciprocal() / q_ref.reciprocal().sum()
    heat = kernel.heat_kernel(
        Xref,
        sources,
        mode="density",
        steps=[1, 2, 3],
        alpha=1.0,
        location_weights=weights,
    )
    mass = (heat * weights[None, :, None]).sum(dim=1)

    assert heat.shape == (3, 7, 3)
    assert torch.all(torch.isfinite(heat))
    assert torch.allclose(mass, torch.ones_like(mass))


def test_scdm_uniform_heat_kernel_direct_reference_matches_symmetric_sections():
    Xref = torch.linspace(0.0, 1.0, 12, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.08], [0.34], [0.77]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.04,
        alpha_init=0.5,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    heat = kernel.heat_kernel(Xref, sources, mode="uniform", steps=1, mass_normalization="none")
    expected = Xref.shape[0] * kernel(sources, Xref).T

    assert heat.shape == (12, 3)
    assert torch.allclose(heat, expected)


def test_scdm_uniform_heat_kernel_normalizes_each_source_mass():
    Xref = torch.linspace(0.0, 1.0, 18, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.08], [0.34], [0.77]], dtype=torch.float64)
    locations = torch.linspace(0.0, 1.0, 27, dtype=torch.float64)[:, None]
    weights = torch.full((27,), 1.0 / 27.0, dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.03,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    heat = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=[1, 2, 4],
        location_weights=weights,
    )
    mass = (heat * weights[None, :, None]).sum(dim=1)

    assert heat.shape == (3, 27, 3)
    assert torch.all(torch.isfinite(heat))
    assert torch.allclose(mass, torch.ones_like(mass))


def _embedded_circle_points(n_points: int) -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * torch.pi, n_points + 1, dtype=torch.float64)[:-1]
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)


def test_scdm_estimates_uniform_reference_volume_with_diagnostics():
    Xref = _embedded_circle_points(512)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.00125, alpha_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    volume, diagnostics = kernel.estimate_reference_volume(
        1,
        warn=False,
        return_diagnostics=True,
    )

    assert torch.isclose(volume, torch.tensor(2.0 * torch.pi, dtype=torch.float64), rtol=0.02)
    assert diagnostics["dim"] == 1
    assert diagnostics["method"] == "median"
    assert diagnostics["row_sum_cv"] < 0.05
    assert diagnostics["row_sum_p95_p05"] < 1.1


def test_scdm_estimated_volume_heat_kernel_uses_physical_scale():
    Xref = _embedded_circle_points(256)
    locations = _embedded_circle_points(128)
    sources = _embedded_circle_points(4)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.00125, alpha_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(Xref)
    volume = kernel.estimate_reference_volume(1, warn=False)

    heat = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=2,
        volume_normalization="estimate_volume",
        volume_dim=1,
        volume_estimate_warnings=False,
    )

    assert torch.allclose(heat.mean(dim=0), volume.reciprocal().expand(sources.shape[0]))


def test_scdm_estimated_volume_heat_kernel_returns_diagnostics():
    Xref = _embedded_circle_points(256)
    locations = _embedded_circle_points(128)
    sources = _embedded_circle_points(4)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.00125, alpha_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    heat, diagnostics = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=2,
        volume_normalization="estimate_volume",
        volume_dim=1,
        volume_estimate_warnings=False,
        return_diagnostics=True,
    )

    assert diagnostics["volume_normalization"] == "estimate_volume"
    assert diagnostics["dim"] == 1
    assert abs(float(diagnostics["volume"]) - 2.0 * torch.pi) < 0.2
    assert torch.allclose(
        heat.mean(dim=0),
        torch.full(
            (sources.shape[0],),
            1.0 / float(diagnostics["volume"]),
            dtype=torch.float64,
        ),
    )


def test_scdm_explicit_volume_heat_kernel_uses_physical_scale():
    Xref = _embedded_circle_points(64)
    locations = _embedded_circle_points(32)
    sources = _embedded_circle_points(3)
    volume = torch.tensor(2.0 * torch.pi, dtype=torch.float64)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.01, alpha_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    heat = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=1,
        volume_normalization="explicit_volume",
        volume=volume,
    )

    assert torch.allclose(heat.mean(dim=0), volume.reciprocal().expand(sources.shape[0]))


def test_scdm_volume_normalization_rejects_explicit_location_weights():
    Xref = _embedded_circle_points(32)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.01, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    with pytest.raises(ValueError, match="location_weights"):
        kernel.heat_kernel(
            Xref,
            Xref[:2],
            mode="uniform",
            location_weights=torch.full((32,), 1.0 / 32.0, dtype=torch.float64),
            volume_normalization="explicit_volume",
            volume=1.0,
        )


def test_scdm_volume_estimate_warns_for_nonuniform_row_sums():
    Xref = torch.tensor(
        [[0.00], [0.01], [0.02], [0.03], [0.50], [0.80], [0.95]], dtype=torch.float64
    )
    kernel = KernelScDMHeat(in_dim=1, eps_init=0.01, dtype=torch.float64, metric="periodic")
    kernel.set_reference_data(Xref)

    with pytest.warns(RuntimeWarning, match="Reference row sums vary"):
        kernel.estimate_reference_volume(1, row_sum_cv_warn=0.01)


def test_scdm_density_heat_kernel_median_normalization_uses_global_source_scale():
    Xref = torch.tensor(
        [[0.00], [0.02], [0.04], [0.18], [0.46], [0.75], [0.97]], dtype=torch.float64
    )
    sources = torch.tensor([[0.01], [0.22], [0.55], [0.91]], dtype=torch.float64)
    locations = torch.linspace(0.0, 1.0, 19, dtype=torch.float64)[:, None]
    weights = torch.full((19,), 1.0 / 19.0, dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.04,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    raw = kernel.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=2,
        location_weights=weights,
        mass_normalization="none",
    )
    normalized = kernel.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=2,
        location_weights=weights,
        mass_normalization="median",
    )
    raw_mass = (raw * weights[:, None]).sum(dim=0)
    normalized_mass = (normalized * weights[:, None]).sum(dim=0)

    assert torch.allclose(normalized, raw / torch.median(raw_mass))
    assert torch.allclose(torch.median(normalized_mass), torch.tensor(1.0, dtype=torch.float64))


def test_scdm_uniform_heat_kernel_median_normalization_uses_global_source_scale():
    Xref = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.02], [0.31], [0.55], [0.89]], dtype=torch.float64)
    locations = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)[:, None]
    weights = torch.full((21,), 1.0 / 21.0, dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.04,
        alpha_init=0.5,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    raw = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=2,
        location_weights=weights,
        mass_normalization="none",
    )
    normalized = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=2,
        location_weights=weights,
        mass_normalization="median",
    )
    raw_mass = (raw * weights[:, None]).sum(dim=0)
    normalized_mass = (normalized * weights[:, None]).sum(dim=0)

    assert torch.allclose(normalized, raw / torch.median(raw_mass))
    assert torch.allclose(torch.median(normalized_mass), torch.tensor(1.0, dtype=torch.float64))


def test_scdm_heat_kernel_multiple_steps_preserve_requested_order():
    Xref = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.12], [0.62]], dtype=torch.float64)
    locations = torch.tensor([[0.18], [0.48], [0.81]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.05,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    stacked = kernel.heat_kernel(locations, sources, mode="uniform", steps=[3, 1, 2])

    assert stacked.shape == (3, 3, 2)
    assert torch.allclose(
        stacked[0], kernel.heat_kernel(locations, sources, mode="uniform", steps=3)
    )
    assert torch.allclose(
        stacked[1], kernel.heat_kernel(locations, sources, mode="uniform", steps=1)
    )
    assert torch.allclose(
        stacked[2], kernel.heat_kernel(locations, sources, mode="uniform", steps=2)
    )


def test_scdm_heat_kernel_supports_broadcast_batches():
    Xref = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)[:, None]
    sources = torch.tensor([[[0.08], [0.25]], [[0.45], [0.70]]], dtype=torch.float64)
    locations = torch.tensor([[[0.15], [0.35], [0.90]]], dtype=torch.float64)
    kernel = KernelScDMHeat(
        in_dim=1,
        eps_init=0.05,
        alpha_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    heat = kernel.heat_kernel(locations, sources, mode="density", steps=2)

    assert heat.shape == (2, 3, 2)
    assert torch.all(torch.isfinite(heat))


def test_scdm_heat_kernel_validates_mode_steps_and_alpha():
    Xref = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)[:, None]
    kernel = KernelScDMHeat(in_dim=1, eps_init=0.03, dtype=torch.float64)
    kernel.set_reference_data(Xref)

    with pytest.raises(ValueError, match="positive"):
        kernel.heat_kernel(Xref, steps=0)
    with pytest.raises(ValueError, match="mode"):
        kernel.heat_kernel(Xref, mode="unknown")
    with pytest.raises(ValueError, match="alpha"):
        kernel.heat_kernel(Xref, mode="uniform", alpha=1.0)
    with pytest.raises(ValueError, match="mass_normalization"):
        kernel.heat_kernel(Xref, mode="uniform", mass_normalization="bad")
    with pytest.raises(ValueError, match="location_weights"):
        kernel.heat_kernel(
            Xref,
            mode="uniform",
            location_weights=torch.ones(Xref.shape[0] + 1, dtype=torch.float64),
        )


def test_scdm_heat_kernel_keeps_section_helpers_private():
    kernel = KernelScDM(in_dim=1, eps_init=0.03, dtype=torch.float64)

    assert not hasattr(kernel, "markov_sections")
    assert not hasattr(kernel, "density_sections")
    assert not hasattr(kernel, "volume_weights")
    assert not hasattr(kernel, "heat_diagnostics")
    assert not hasattr(kernel, "row_sums")
    assert not hasattr(kernel, "reference_row_sums")
    assert not hasattr(kernel, "raw_kernel")
    assert not hasattr(kernel, "heat_kernel")
    assert not hasattr(kernel, "estimate_reference_volume")
    assert not hasattr(kernel, "density_eps")
    assert not hasattr(kernel, "density_bandwidth_factor")


def _nonuniform_euclidean_reference() -> torch.Tensor:
    return torch.tensor(
        [
            [0.00, 0.00],
            [0.04, 0.01],
            [0.10, 0.02],
            [0.25, 0.08],
            [0.45, 0.18],
            [0.72, 0.37],
            [0.91, 0.62],
            [1.00, 0.95],
        ],
        dtype=torch.float64,
    )


def _dense_keops_scdm_pair(
    *,
    eps_init: float = 0.08,
    alpha_init: float = 1.0,
    density_bandwidth_factor: float = 1.4,
) -> tuple[KernelScDMHeat, KernelScDMHeat]:
    xref = _nonuniform_euclidean_reference()
    dense = KernelScDMHeat(
        in_dim=2,
        eps_init=eps_init,
        alpha_init=alpha_init,
        dtype=torch.float64,
        density_bandwidth_factor=density_bandwidth_factor,
    )
    keops = KernelScDMHeat(
        in_dim=2,
        eps_init=eps_init,
        alpha_init=alpha_init,
        dtype=torch.float64,
        density_bandwidth_factor=density_bandwidth_factor,
        backend="keops",
    )
    dense.set_reference_data(xref)
    keops.set_reference_data(xref)
    return dense, keops


def test_keops_scdm_backend_matches_dense_euclidean_sections():
    pytest.importorskip("pykeops")
    Xref = _embedded_circle_points(32)
    locations = _embedded_circle_points(24)
    sources = _embedded_circle_points(3)
    dense = KernelScDMHeat(in_dim=2, eps_init=0.2, alpha_init=1.0, dtype=torch.float64)
    keops = KernelScDMHeat(
        in_dim=2,
        eps_init=0.2,
        alpha_init=1.0,
        dtype=torch.float64,
        backend="keops",
    )
    dense.set_reference_data(Xref)
    keops.set_reference_data(Xref)

    dense_heat = dense.heat_kernel(locations, sources, mode="uniform", steps=[1, 3])
    keops_heat = keops.heat_kernel(locations, sources, mode="uniform", steps=[1, 3])

    assert keops_heat.shape == dense_heat.shape
    assert torch.allclose(keops_heat, dense_heat, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["uniform", "density"])
def test_keops_scdm_batched_sources_singleton_locations_match_dense(mode: str):
    pytest.importorskip("pykeops")
    sources = torch.tensor(
        [
            [[0.03, 0.01], [0.38, 0.15], [0.88, 0.58]],
            [[0.05, 0.03], [0.44, 0.20], [0.94, 0.70]],
        ],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [[[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50]]],
        dtype=torch.float64,
    )
    dense, keops = _dense_keops_scdm_pair()
    kwargs = {"alpha": 0.75} if mode == "density" else {}

    expected = dense.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )
    actual = keops.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["uniform", "density"])
def test_keops_scdm_unbatched_sources_batched_locations_match_dense(mode: str):
    pytest.importorskip("pykeops")
    sources = torch.tensor([[0.03, 0.01], [0.38, 0.15], [0.88, 0.58]], dtype=torch.float64)
    locations = torch.tensor(
        [
            [[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50]],
            [[0.06, 0.03], [0.25, 0.10], [0.61, 0.28], [0.91, 0.62]],
        ],
        dtype=torch.float64,
    )
    dense, keops = _dense_keops_scdm_pair()
    kwargs = {"alpha": 0.75} if mode == "density" else {}

    expected = dense.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )
    actual = keops.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["uniform", "density"])
def test_keops_scdm_multi_leading_batch_heat_kernel_matches_dense(mode: str):
    pytest.importorskip("pykeops")
    sources = torch.tensor(
        [
            [[[0.03, 0.01], [0.38, 0.15], [0.88, 0.58]]],
            [[[0.05, 0.03], [0.44, 0.20], [0.94, 0.70]]],
        ],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [
            [
                [[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50]],
                [[0.06, 0.03], [0.25, 0.10], [0.61, 0.28], [0.91, 0.62]],
                [[0.10, 0.05], [0.30, 0.12], [0.68, 0.32], [0.96, 0.80]],
            ]
        ],
        dtype=torch.float64,
    )
    dense, keops = _dense_keops_scdm_pair()
    kwargs = {"alpha": 0.75} if mode == "density" else {}

    expected = dense.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )
    actual = keops.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )

    assert actual.shape == expected.shape
    assert actual.shape[1:] == (2, 3, 4, 3)
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["uniform", "density"])
def test_keops_scdm_batched_sources_none_matches_dense(mode: str):
    pytest.importorskip("pykeops")
    locations = torch.tensor(
        [
            [[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50]],
            [[0.06, 0.03], [0.25, 0.10], [0.61, 0.28], [0.91, 0.62]],
        ],
        dtype=torch.float64,
    )
    dense, keops = _dense_keops_scdm_pair()
    kwargs = {"alpha": 0.75} if mode == "density" else {}

    expected = dense.heat_kernel(
        locations,
        sources=None,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )
    actual = keops.heat_kernel(
        locations,
        sources=None,
        mode=mode,
        steps=[1, 2, 3],
        mass_normalization="none",
        **kwargs,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["uniform", "density"])
def test_keops_scdm_batched_location_weights_match_dense(mode: str):
    pytest.importorskip("pykeops")
    sources = torch.tensor(
        [
            [[[0.03, 0.01], [0.38, 0.15], [0.88, 0.58]]],
            [[[0.05, 0.03], [0.44, 0.20], [0.94, 0.70]]],
        ],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [
            [
                [[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50]],
                [[0.06, 0.03], [0.25, 0.10], [0.61, 0.28], [0.91, 0.62]],
                [[0.10, 0.05], [0.30, 0.12], [0.68, 0.32], [0.96, 0.80]],
            ]
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.18, 0.22, 0.26, 0.34], [0.25, 0.25, 0.25, 0.25]]],
        dtype=torch.float64,
    )
    dense, keops = _dense_keops_scdm_pair()
    kwargs = {"alpha": 0.75} if mode == "density" else {}

    expected = dense.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        location_weights=weights,
        **kwargs,
    )
    actual = keops.heat_kernel(
        locations,
        sources,
        mode=mode,
        steps=[1, 2, 3],
        location_weights=weights,
        **kwargs,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_keops_scdm_volume_estimate_diagnostics_match_dense_contract():
    pytest.importorskip("pykeops")
    Xref = _embedded_circle_points(64)
    locations = _embedded_circle_points(32)
    sources = _embedded_circle_points(2)
    kernel = KernelScDMHeat(in_dim=2, eps_init=0.05, dtype=torch.float64, backend="keops")
    kernel.set_reference_data(Xref)

    heat, diagnostics = kernel.heat_kernel(
        locations,
        sources,
        mode="uniform",
        steps=2,
        volume_normalization="estimate_volume",
        volume_dim=1,
        volume_estimate_warnings=False,
        return_diagnostics=True,
    )

    assert heat.shape == (32, 2)
    assert diagnostics["volume_normalization"] == "estimate_volume"
    assert diagnostics["dim"] == 1
    assert float(diagnostics["volume"]) > 0.0


def test_keops_scdm_density_heat_kernel_matches_dense_euclidean_sections():
    pytest.importorskip("pykeops")
    Xref = torch.tensor(
        [
            [0.00, 0.00],
            [0.04, 0.01],
            [0.10, 0.02],
            [0.25, 0.08],
            [0.45, 0.18],
            [0.72, 0.37],
            [0.91, 0.62],
            [1.00, 0.95],
        ],
        dtype=torch.float64,
    )
    locations = torch.tensor(
        [[0.02, 0.02], [0.21, 0.06], [0.52, 0.23], [0.87, 0.50], [0.96, 0.87]],
        dtype=torch.float64,
    )
    sources = torch.tensor([[0.03, 0.01], [0.38, 0.15], [0.88, 0.58]], dtype=torch.float64)
    dense = KernelScDMHeat(
        in_dim=2,
        eps_init=0.08,
        alpha_init=1.0,
        dtype=torch.float64,
        density_bandwidth_factor=1.4,
    )
    keops = KernelScDMHeat(
        in_dim=2,
        eps_init=0.08,
        alpha_init=1.0,
        dtype=torch.float64,
        density_bandwidth_factor=1.4,
        backend="keops",
    )
    dense.set_reference_data(Xref)
    keops.set_reference_data(Xref)

    dense_heat = dense.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=[1, 2, 4],
        alpha=0.75,
        mass_normalization="none",
    )
    keops_heat = keops.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=[1, 2, 4],
        alpha=0.75,
        mass_normalization="none",
    )

    assert keops_heat.shape == dense_heat.shape
    assert torch.allclose(keops_heat, dense_heat, rtol=1e-10, atol=1e-10)


def test_keops_scdm_density_heat_kernel_mass_normalization_matches_dense():
    pytest.importorskip("pykeops")
    Xref = torch.tensor(
        [[0.0], [0.02], [0.05], [0.20], [0.38], [0.61], [0.82], [0.97]],
        dtype=torch.float64,
    )
    locations = torch.tensor([[0.01], [0.18], [0.41], [0.73], [0.93]], dtype=torch.float64)
    sources = torch.tensor([[0.03], [0.35], [0.90]], dtype=torch.float64)
    weights = torch.tensor([0.08, 0.14, 0.20, 0.25, 0.33], dtype=torch.float64)
    dense = KernelScDMHeat(in_dim=1, eps_init=0.07, dtype=torch.float64)
    keops = KernelScDMHeat(in_dim=1, eps_init=0.07, dtype=torch.float64, backend="keops")
    dense.set_reference_data(Xref)
    keops.set_reference_data(Xref)

    dense_heat = dense.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=[1, 3],
        location_weights=weights,
    )
    keops_heat = keops.heat_kernel(
        locations,
        sources,
        mode="density",
        steps=[1, 3],
        location_weights=weights,
    )

    assert keops_heat.shape == dense_heat.shape
    assert torch.allclose(keops_heat, dense_heat, rtol=1e-10, atol=1e-10)


def test_keops_scdm_backend_rejects_non_euclidean_metric():
    with pytest.raises(NotImplementedError, match="Euclidean"):
        kernel = KernelScDMHeat(in_dim=1, eps_init=0.1, metric="periodic", backend="keops")
        kernel.set_reference_data(torch.linspace(0.0, 1.0, 4, dtype=torch.float64)[:, None])
