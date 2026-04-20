import numpy as np
import pytest

from dymad.sako.snapshot import build_spectral_snapshot


def test_build_spectral_snapshot_with_full_weights():
    p0 = np.ones((8, 4))
    p1 = np.zeros((8, 4))
    w = (np.eye(4),)

    snapshot = build_spectral_snapshot(
        model_class="DKBF",
        checkpoint_path="checkpoints/model.pt",
        encoded_p0=p0,
        encoded_p1=p1,
        weights=w,
        input_dim=2,
        obs_dim=4,
        metadata={"processor_mode": "full"},
    )

    assert snapshot.sample_count == 8
    assert snapshot.koopman_weights.mode == "full"
    assert snapshot.koopman_weights.full_matrix is not None
    assert snapshot.koopman_weights.left_factor is None
    assert snapshot.metadata["processor_mode"] == "full"


def test_build_spectral_snapshot_with_low_rank_weights():
    p0 = np.ones((5, 3))
    p1 = np.zeros((5, 3))
    u = np.ones((3, 2))
    v = np.ones((3, 2))

    snapshot = build_spectral_snapshot(
        model_class="DKBF",
        checkpoint_path="checkpoints/model.pt",
        encoded_p0=p0,
        encoded_p1=p1,
        weights=(u, v),
        input_dim=2,
        obs_dim=3,
        metadata={"processor_mode": "factorized"},
    )

    assert snapshot.sample_count == 5
    assert snapshot.koopman_weights.mode == "low_rank"
    assert snapshot.koopman_weights.full_matrix is None
    assert snapshot.koopman_weights.left_factor is not None
    assert snapshot.koopman_weights.right_factor is not None


def test_build_spectral_snapshot_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        build_spectral_snapshot(
            model_class="DKBF",
            checkpoint_path="checkpoints/model.pt",
            encoded_p0=np.ones((4, 2)),
            encoded_p1=np.ones((3, 2)),
            weights=(np.eye(2),),
            input_dim=2,
            obs_dim=2,
        )


def test_build_spectral_snapshot_rejects_invalid_weight_arity():
    with pytest.raises(ValueError, match="either one full matrix or two low-rank factors"):
        build_spectral_snapshot(
            model_class="DKBF",
            checkpoint_path="checkpoints/model.pt",
            encoded_p0=np.ones((4, 2)),
            encoded_p1=np.ones((4, 2)),
            weights=(np.eye(2), np.eye(2), np.eye(2)),
            input_dim=2,
            obs_dim=2,
        )
