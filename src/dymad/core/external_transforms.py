"""External and approximate-gradient transform modules."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

import numpy as np
import sklearn.manifold as skm

from dymad.core.transform_module import ExternalTransformModule
from dymad.numerics import DM, DMF, VBDM, Manifold, ManifoldAltTree, complex_step


class CallableExternalTransform(ExternalTransformModule):
    """External wrapper for callable lift-like transforms."""

    def __init__(self, fobs=None, finv=None, **kwargs) -> None:
        super().__init__(invertibility="approximate", supports_gradients="approximate")
        self.fobs = fobs
        self.finv = finv
        self.kwargs = kwargs
        self._pseudo_inverse_matrix: np.ndarray | None = None

    def _fit_external(self, data: list[np.ndarray]) -> None:
        if not data:
            return
        self.input_dim = int(data[0].shape[-1])
        sample = self._apply_forward(data[0])
        self.output_dim = int(sample.shape[-1])
        if self.finv is None:
            outputs = [self._apply_forward(item) for item in data]
            self._pseudo_inverse_matrix = np.linalg.lstsq(
                np.vstack(outputs),
                np.vstack(data),
                rcond=None,
            )[0]

    def _forward_external(self, data: list[np.ndarray]) -> list[np.ndarray]:
        return [self._apply_forward(item) for item in data]

    def _inverse_external(self, data: list[np.ndarray]) -> list[np.ndarray]:
        return [self._apply_inverse(item) for item in data]

    def _forward_jacobian_external(self, ref: np.ndarray) -> np.ndarray:
        def func(x):
            return self._apply_forward(x.reshape(1, -1)).reshape(-1)

        return complex_step(func, np.asarray(ref).reshape(-1))

    def _inverse_modes_external(self, ref: np.ndarray) -> np.ndarray:
        if self.finv is None and self._pseudo_inverse_matrix is not None:
            return self._pseudo_inverse_matrix.T

        def func(x):
            return self._apply_inverse(x.reshape(1, -1)).reshape(-1)

        return complex_step(func, np.asarray(ref).reshape(-1))

    def _apply_forward(self, data: np.ndarray) -> np.ndarray:
        if not callable(self.fobs):
            raise ValueError("CallableExternalTransform requires a callable forward observable.")
        input_dim = self.input_dim
        if input_dim is None:
            raise ValueError("CallableExternalTransform input_dim is not initialized.")
        return np.asarray(self.fobs(data.reshape(-1, input_dim), **self.kwargs))

    def _apply_inverse(self, data: np.ndarray) -> np.ndarray:
        output_dim = self.output_dim
        if output_dim is None:
            raise ValueError("CallableExternalTransform output_dim is not initialized.")
        if callable(self.finv):
            return np.asarray(self.finv(data.reshape(-1, output_dim), **self.kwargs))
        if self._pseudo_inverse_matrix is None:
            raise ValueError("CallableExternalTransform inverse parameters are not initialized.")
        return data.reshape(-1, output_dim).dot(self._pseudo_inverse_matrix)


class _NDRTransformBase(ExternalTransformModule):
    """Shared external wrapper for manifold-learning transforms."""

    def __init__(self, **kwargs) -> None:
        super().__init__(invertibility="approximate", supports_gradients="approximate")
        self.embedding_dim = kwargs.pop("edim", None)
        self.inverse_mode = kwargs.pop("inverse", None)
        self.knn = kwargs.pop("Knn", None)
        self.kphi = kwargs.pop("Kphi", None)
        self.order = kwargs.pop("order", None)
        self.rcond = kwargs.pop("rcond", None)

        self._ndr: Any = None
        self._pseudo_inverse_matrix: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._Z: np.ndarray | None = None
        self._man_bck: Any = None
        self._man_for: Any = None

    def _fit_external(self, data: list[np.ndarray]) -> None:
        self._make_ndr()
        if self._ndr is None:
            raise ValueError(f"{type(self).__name__} failed to initialize its NDR model.")
        X = np.vstack(data)
        self._X = X
        self.input_dim = int(X.shape[-1])
        Z = np.asarray(self._ndr.fit_transform(X))
        self._Z = Z
        self.output_dim = int(Z.shape[-1])
        self._prepare_inverse()

    def _forward_external(self, data: list[np.ndarray]) -> list[np.ndarray]:
        if self._ndr is None:
            raise ValueError(f"{type(self).__name__} must be fitted before transform(...).")
        return [np.asarray(self._ndr.transform(np.atleast_2d(item))) for item in data]

    def _inverse_external(self, data: list[np.ndarray]) -> list[np.ndarray]:
        inverse_mode = str(self.inverse_mode).lower()
        if inverse_mode == "pinv":
            assert self._pseudo_inverse_matrix is not None
            return [item.dot(self._pseudo_inverse_matrix) for item in data]
        if inverse_mode == "gmls":
            assert self._man_bck is not None and self._X is not None
            return [cast(np.ndarray, self._man_bck.gmls(item, self._X)) for item in data]
        raise ValueError(f"Unknown inverse mode {self.inverse_mode}")

    def _forward_jacobian_external(self, ref: np.ndarray) -> np.ndarray:
        if self._man_for is None or self._Z is None:
            raise ValueError("Forward modes are only defined after the transform is fitted.")
        return self._man_for.gmls(ref, self._Z, ret_der=True)[1]

    def _inverse_modes_external(self, ref: np.ndarray) -> np.ndarray:
        if self._pseudo_inverse_matrix is not None:
            return self._pseudo_inverse_matrix
        if self._man_bck is None or self._X is None:
            raise ValueError("Backward modes are only defined after the transform is fitted.")
        modes = self._man_bck.gmls(ref, self._X, ret_der=True)[1]
        return np.swapaxes(modes, -2, -1)

    def _prepare_inverse(self) -> None:
        inverse_mode = str(self.inverse_mode).lower()
        if inverse_mode == "gmls":
            assert self._Z is not None and self._X is not None
            self._man_bck = Manifold(
                self._Z,
                self.order,
                K=self.knn,
                g=self.kphi,
                T=self.kphi,
            )
            self._man_for = ManifoldAltTree(
                self._X,
                self.order,
                K=self.knn,
                g=self.kphi,
                T=self.kphi,
                tree_data=self._Z,
                tree_transform=lambda x: np.asarray(self._ndr.transform(np.atleast_2d(x))),
            )
            self._pseudo_inverse_matrix = None
            return
        if inverse_mode == "pinv":
            assert self._Z is not None and self._X is not None
            self._pseudo_inverse_matrix = np.linalg.pinv(
                self._Z,
                rcond=self.rcond,
                hermitian=False,
            ).dot(self._X)
            self._man_bck = None
            self._man_for = None
            return
        raise ValueError(f"Unknown inverse mode {self.inverse_mode}")

    @abstractmethod
    def _make_ndr(self) -> None:
        raise NotImplementedError


class IsomapTransform(_NDRTransformBase):
    def _make_ndr(self) -> None:
        self._ndr = skm.Isomap(n_neighbors=self.knn, n_components=self.embedding_dim)


class DiffMapTransform(_NDRTransformBase):
    def __init__(self, **kwargs) -> None:
        self.alpha = kwargs.get("alpha", 1)
        self.epsilon = kwargs.get("epsilon", None)
        self.mode = kwargs.get("mode", "full")
        super().__init__(**kwargs)

    def _make_ndr(self) -> None:
        if self.mode == "full":
            self._ndr = DMF(n_components=self.embedding_dim, alpha=self.alpha, epsilon=self.epsilon)
        else:
            self._ndr = DM(
                n_components=self.embedding_dim,
                n_neighbors=self.knn,
                alpha=self.alpha,
                epsilon=self.epsilon,
            )


class DiffMapVBTransform(DiffMapTransform):
    def __init__(self, **kwargs) -> None:
        self.kb = kwargs.get("Kb", None)
        super().__init__(**kwargs)

    def _make_ndr(self) -> None:
        self._ndr = VBDM(
            n_neighbors=self.knn,
            n_components=self.embedding_dim,
            Kb=self.kb,
            operator="lb",
        )
