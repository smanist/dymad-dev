import pytest
import torch
import torch.nn as nn

from dymad.modules import KernelScDM, KRRMultiOutputShared


def test_scdm_forward_preserves_symmetric_normalized_kernel():
    X = torch.tensor([[0.0, 0.0], [0.25, 0.5], [0.5, 0.25], [1.0, 1.0]], dtype=torch.float64)
    kernel = KernelScDM(in_dim=2, eps_init=0.2, t_init=1.0, dtype=torch.float64)
    kernel.set_reference_data(X)

    W = torch.exp(-(torch.cdist(X, X) ** 2) / (4.0 * kernel.eps))
    q = W.sum(dim=-1)
    D = q ** (-kernel.t)
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
    ).raw_kernel(X, Z, eps=eps)
    periodic = KernelScDM(
        in_dim=1, eps_init=0.1, metric="periodic", dtype=torch.float64
    ).raw_kernel(X, Z, eps=eps)
    periodic_axis_subset = KernelScDM(
        in_dim=2,
        eps_init=0.1,
        metric="periodic",
        periodic_axes=(0,),
        dtype=torch.float64,
    ).raw_kernel(
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


def test_scdm_density_sections_have_unit_reference_weight_mass():
    Xref = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)[:, None]
    sources = torch.tensor([[0.02], [0.37], [0.89]], dtype=torch.float64)
    kernel = KernelScDM(
        in_dim=1,
        eps_init=0.01,
        t_init=1.0,
        dtype=torch.float64,
        metric="periodic",
        density_bandwidth_factor=1.5,
    )
    kernel.set_reference_data(Xref)

    density = kernel.density_sections(sources)
    weights = kernel.volume_weights()
    mass = (density * weights).sum(dim=1)

    assert density.shape == (3, 24)
    assert weights.shape == (24,)
    assert torch.all(torch.isfinite(density))
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(mass, torch.ones_like(mass))


def test_scdm_density_sections_are_finite_for_nonuniform_reference_points():
    Xref = torch.tensor(
        [[0.00], [0.01], [0.015], [0.20], [0.55], [0.73], [0.95]], dtype=torch.float64
    )
    sources = torch.tensor([[0.02], [0.40], [0.98]], dtype=torch.float64)
    kernel = KernelScDM(
        in_dim=1,
        eps_init=0.02,
        t_init=1.0,
        dtype=torch.float64,
        metric="periodic",
    )
    kernel.set_reference_data(Xref)

    density = kernel.density_sections(sources, alpha=1.0)
    weights = kernel.volume_weights()
    mass = (density * weights).sum(dim=1)

    assert torch.all(torch.isfinite(density))
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(mass, torch.ones_like(mass))


def test_krr_reference_sections_work_without_fit_for_shared_kernel():
    Xref = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)[:, None]
    Xnew = torch.tensor([[0.05], [0.45]], dtype=torch.float64)
    model = KRRMultiOutputShared(
        kernel=KernelScDM(
            in_dim=1,
            eps_init=0.02,
            dtype=torch.float64,
            metric="periodic",
        ),
        ridge_init=0.0,
    )

    model.set_reference_data(Xref)

    density = model.kernel_sections(Xnew, mode="density")
    weights = model.reference_weights()
    assert density.shape == (2, 16)
    assert weights.shape == (16,)
    assert torch.allclose((density * weights).sum(dim=1), torch.ones(2, dtype=torch.float64))

    with pytest.raises(RuntimeError):
        model(Xnew)


def test_krr_reference_sections_reject_module_list_kernels():
    Xref = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)[:, None]
    model = KRRMultiOutputShared(
        kernel=KernelScDM(in_dim=1, eps_init=0.02, dtype=torch.float64),
        ridge_init=0.0,
    )
    model.kernel = nn.ModuleList(
        [
            KernelScDM(in_dim=1, eps_init=0.02, dtype=torch.float64),
            KernelScDM(in_dim=1, eps_init=0.03, dtype=torch.float64),
        ]
    )
    model.set_reference_data(Xref)

    with pytest.raises(TypeError, match="single kernel"):
        model.kernel_sections(Xref)
