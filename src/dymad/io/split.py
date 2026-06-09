from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from dymad.core import ScalerTransform
from dymad.core.transform_module import TransformModule


@dataclass(frozen=True)
class Split:
    """Reusable supervised array split with fitted x/y transforms."""

    x_train_raw: np.ndarray
    y_train_raw: np.ndarray
    x_val_raw: np.ndarray
    y_val_raw: np.ndarray
    x_test_raw: np.ndarray
    y_test_raw: np.ndarray
    x_transform: TransformModule
    y_transform: TransformModule

    @classmethod
    def from_arrays(
        cls,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_transform: TransformModule | None = None,
        y_transform: TransformModule | None = None,
    ) -> Split:
        x_fit = _fit_array_transform(x_transform or ScalerTransform("std"), [x_train])
        y_fit = _fit_array_transform(y_transform or ScalerTransform("std"), [y_train])
        return cls(
            x_train_raw=np.asarray(x_train),
            y_train_raw=np.asarray(y_train),
            x_val_raw=np.asarray(x_val),
            y_val_raw=np.asarray(y_val),
            x_test_raw=np.asarray(x_test),
            y_test_raw=np.asarray(y_test),
            x_transform=x_fit,
            y_transform=y_fit,
        )

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        return self.x_transform.transform([values])[0]

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        return self.y_transform.transform([values])[0]

    def inverse_x(self, values: np.ndarray) -> np.ndarray:
        return self.x_transform.inverse_transform([values])[0]

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        return self.y_transform.inverse_transform([values])[0]

    @property
    def x_train(self) -> np.ndarray:
        return self.transform_x(self.x_train_raw)

    @property
    def y_train(self) -> np.ndarray:
        return self.transform_y(self.y_train_raw)

    @property
    def x_val(self) -> np.ndarray:
        return self.transform_x(self.x_val_raw)

    @property
    def y_val(self) -> np.ndarray:
        return self.transform_y(self.y_val_raw)

    @property
    def x_test(self) -> np.ndarray:
        return self.transform_x(self.x_test_raw)

    @property
    def y_test(self) -> np.ndarray:
        return self.transform_y(self.y_test_raw)


def _fit_array_transform(
    transform: TransformModule, arrays: Sequence[np.ndarray]
) -> TransformModule:
    transform.fit([torch.as_tensor(item) for item in arrays])
    return transform
