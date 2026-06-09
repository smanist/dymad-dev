from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ValidationFold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class HoldoutValidationPolicy:
    validation_fraction: float = 0.25
    shuffle: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")


@dataclass(frozen=True)
class KFoldValidationPolicy:
    k: int = 4
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("k must be at least 2")


@dataclass(frozen=True)
class TrainValidCountPolicy:
    validation_fraction: float | None = None
    validation_size: int | None = None
    shuffle: bool = False

    def __post_init__(self) -> None:
        if (self.validation_fraction is None) == (self.validation_size is None):
            raise ValueError("provide exactly one of validation_fraction or validation_size")
        if self.validation_fraction is not None and self.validation_fraction <= 0.0:
            raise ValueError("validation_fraction must be positive when provided")
        if self.validation_size is not None and self.validation_size <= 0:
            raise ValueError("validation_size must be positive when provided")


ValidationPolicy = HoldoutValidationPolicy | KFoldValidationPolicy | TrainValidCountPolicy


@dataclass(frozen=True)
class NestedResamplingPolicy:
    mode: Literal["nested_fixed_test"] = "nested_fixed_test"
    test_size: int = 128
    validation: ValidationPolicy = HoldoutValidationPolicy()
    seed: int = 0
    dev_pool_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode != "nested_fixed_test":
            raise ValueError("NestedResamplingPolicy.mode must be 'nested_fixed_test'")
        if self.test_size <= 0:
            raise ValueError("test_size must be positive")
        if self.dev_pool_size is not None and self.dev_pool_size <= 0:
            raise ValueError("dev_pool_size must be positive when provided")


@dataclass(frozen=True)
class LevelSamplePlan:
    refinement: int
    pool_indices: tuple[int, ...]
    validation_folds: tuple[ValidationFold, ...]
    refit_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrialSamplePlan:
    trial: int | str
    test_indices: tuple[int, ...]
    dev_ordering: tuple[int, ...]
    levels: dict[int, LevelSamplePlan]


def build_nested_trial_sample_plan(
    policy: NestedResamplingPolicy,
    *,
    refinement_levels: tuple[float | int | str, ...],
    trial: int | str,
) -> TrialSamplePlan:
    levels = tuple(_as_positive_int_level(level) for level in refinement_levels)
    max_level = max(levels)
    required_pool_size = _required_dev_pool_size(policy.validation, max_level)
    dev_pool_size = policy.dev_pool_size or required_pool_size
    if dev_pool_size < required_pool_size:
        raise ValueError("dev_pool_size must cover the largest training level and validation split")
    rng = np.random.default_rng(_trial_seed(policy.seed, trial))
    dev_ordering = tuple(int(item) for item in rng.permutation(dev_pool_size))
    test_indices = tuple(range(policy.test_size))
    level_plans = {
        level: _build_level_sample_plan(
            policy,
            refinement=level,
            pool_indices=_level_pool_indices(policy.validation, dev_ordering, level),
            test_indices=test_indices,
            rng_seed=_trial_seed(policy.seed + 1_000_003 * level, trial),
        )
        for level in levels
    }
    return TrialSamplePlan(
        trial=trial,
        test_indices=test_indices,
        dev_ordering=dev_ordering,
        levels=level_plans,
    )


def _build_level_sample_plan(
    policy: NestedResamplingPolicy,
    *,
    refinement: int,
    pool_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    rng_seed: int,
) -> LevelSamplePlan:
    folds = (
        _holdout_folds(pool_indices, policy.validation, rng_seed)
        if isinstance(policy.validation, HoldoutValidationPolicy)
        else _train_valid_count_folds(pool_indices, policy.validation, refinement, rng_seed)
        if isinstance(policy.validation, TrainValidCountPolicy)
        else _kfold_folds(pool_indices, policy.validation, rng_seed)
    )
    refit_indices = (
        folds[0].train_indices
        if isinstance(policy.validation, TrainValidCountPolicy)
        else pool_indices
    )
    return LevelSamplePlan(
        refinement=refinement,
        pool_indices=pool_indices,
        validation_folds=folds,
        refit_indices=refit_indices,
        test_indices=test_indices,
    )


def _required_dev_pool_size(policy: ValidationPolicy, max_level: int) -> int:
    if isinstance(policy, TrainValidCountPolicy):
        return max_level + _validation_count(policy, max_level)
    return max_level


def _level_pool_indices(
    policy: ValidationPolicy,
    dev_ordering: tuple[int, ...],
    level: int,
) -> tuple[int, ...]:
    if isinstance(policy, TrainValidCountPolicy):
        return dev_ordering[: level + _validation_count(policy, level)]
    return dev_ordering[:level]


def _holdout_folds(
    pool_indices: tuple[int, ...],
    policy: HoldoutValidationPolicy,
    rng_seed: int,
) -> tuple[ValidationFold, ...]:
    n_pool = len(pool_indices)
    if n_pool < 2:
        raise ValueError("holdout validation requires at least two training-pool samples")
    ordered = np.asarray(pool_indices, dtype=int)
    if policy.shuffle:
        rng = np.random.default_rng(rng_seed)
        ordered = ordered[rng.permutation(n_pool)]
    n_val = int(round(n_pool * policy.validation_fraction))
    n_val = min(max(n_val, 1), n_pool - 1)
    train = tuple(int(item) for item in ordered[: n_pool - n_val])
    valid = tuple(int(item) for item in ordered[n_pool - n_val :])
    return (ValidationFold(train_indices=train, validation_indices=valid),)


def _kfold_folds(
    pool_indices: tuple[int, ...],
    policy: KFoldValidationPolicy,
    rng_seed: int,
) -> tuple[ValidationFold, ...]:
    n_pool = len(pool_indices)
    if policy.k > n_pool:
        raise ValueError("k-fold validation requires k <= training-pool sample count")
    ordered = np.asarray(pool_indices, dtype=int)
    if policy.shuffle:
        rng = np.random.default_rng(rng_seed)
        ordered = ordered[rng.permutation(n_pool)]
    folds = []
    for valid in np.array_split(ordered, policy.k):
        valid_set = {int(item) for item in valid}
        train = tuple(int(item) for item in ordered if int(item) not in valid_set)
        folds.append(
            ValidationFold(
                train_indices=train,
                validation_indices=tuple(int(item) for item in valid),
            )
        )
    return tuple(folds)


def _train_valid_count_folds(
    pool_indices: tuple[int, ...],
    policy: TrainValidCountPolicy,
    refinement: int,
    rng_seed: int,
) -> tuple[ValidationFold, ...]:
    n_val = _validation_count(policy, refinement)
    expected = refinement + n_val
    if len(pool_indices) < expected:
        raise ValueError("train-valid count split requires n_train + n_valid samples")
    ordered = np.asarray(pool_indices, dtype=int)
    if policy.shuffle:
        rng = np.random.default_rng(rng_seed)
        ordered = ordered[rng.permutation(len(ordered))]
    train = tuple(int(item) for item in ordered[:refinement])
    valid = tuple(int(item) for item in ordered[refinement : refinement + n_val])
    return (ValidationFold(train_indices=train, validation_indices=valid),)


def _validation_count(policy: TrainValidCountPolicy, n_train: int) -> int:
    if policy.validation_size is not None:
        return int(policy.validation_size)
    assert policy.validation_fraction is not None
    return max(1, int(round(n_train * policy.validation_fraction)))


def _as_positive_int_level(level: float | int | str) -> int:
    try:
        value = int(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"nested resampling requires integer refinement levels, got {level!r}"
        ) from exc
    if str(level) not in {str(value), f"{value}.0"} and float(level) != float(value):
        raise ValueError(f"nested resampling requires integer refinement levels, got {level!r}")
    if value <= 0:
        raise ValueError("nested resampling refinement levels must be positive")
    return value


def _trial_seed(seed: int, trial: int | str) -> int:
    digest = hashlib.blake2b(str(trial).encode("utf-8"), digest_size=8).digest()
    trial_value = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(seed) + trial_value) % (2**63)
