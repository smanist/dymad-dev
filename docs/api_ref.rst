API Reference
=============

The complete public API is listed below, grouped by module.

Kernel Backends And KRR Solvers
-------------------------------

Kernel ``forward()`` remains the explicit materialized block API for diagnostics,
small problems, and dense fallbacks. Scalar and operator-valued kernels also
provide ``apply(X, Z, values)`` for computing ``K(X, Z) @ values`` without
requiring callers to materialize the kernel block when a backend supports a
matrix-free reduction.

KRR uses dense Cholesky by default, which remains the recommended small-problem
path. Opt into matrix-free fitting with ``solver="matrix_free_cg"`` or use
``solver="auto"`` to switch above the materialized-operator threshold. The
matrix-free path uses ``kernel.apply`` for fitting and prediction.

KeOps support is optional and can be installed with ``dymad[keops]``. Use
``backend="keops"`` on ``KernelScRBF``, ``KernelScExp``, or Euclidean
``KernelScDM`` to enable KeOps-backed reductions. Euclidean ``KernelScDM``
supports KeOps-backed uniform and density heat sections with the same
leading-batch shape convention as dense torch for Euclidean inputs. Periodic and
other non-Euclidean diffusion-map kernels are torch-only for matrix-free KeOps
use and raise a clear ``NotImplementedError`` when requested.

.. autosummary::
   :toctree: api
   :recursive:
   :caption: API Modules

   dymad.io
   dymad.kernel_analysis
   dymad.losses
   dymad.models
   dymad.modules
   dymad.numerics
   dymad.sako
   dymad.training
   dymad.utils

See also

.. toctree::
   :maxdepth: 2

   api_maps
