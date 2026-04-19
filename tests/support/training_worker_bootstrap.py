from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import yaml

import dymad.training


class _FakeTrainer:
    def __init__(self, config_path, model_class, config_mod=None, device=None, max_workers=1):
        del model_class, config_mod, max_workers
        self.config_path = Path(config_path)
        self.device = torch.device("cpu") if device is None else device

    def train(self):
        print("fake trainer starting", flush=True)
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        run_name = config["model"]["name"]
        run_root = self.config_path.parent / run_name
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_root / f"{run_name}.pt"
        torch.save(
            {
                "config": config,
                "train_md": {
                    "n_state_features": 2,
                    "n_aux_features": 0,
                    "n_control_features": 1,
                    "n_parameters": 0,
                    "transform_x_state": None,
                    "transform_u_state": None,
                },
                "valid_md": {
                    "n_state_features": 2,
                    "n_aux_features": 0,
                    "n_control_features": 1,
                    "n_parameters": 0,
                    "transform_x_state": None,
                    "transform_u_state": None,
                },
                "model_state_dict": {},
                "best_loss": {"valid_total": 0.1},
                "hist": [{"train_total": [0.2], "valid_total": [0.1]}],
                "crit": [],
                "epoch_times": [0.5],
                "converged": False,
                "device": "cpu",
                "epoch": 1,
            },
            checkpoint_path,
        )
        np.savez_compressed(
            run_root / f"{run_name}_summary.npz",
            model_name=run_name,
            total_training_time=1.0,
            avg_epoch_time=0.5,
            final_train_loss=0.2,
            final_valid_loss=0.1,
            best_valid_loss=np.array({"valid_total": 0.1}, dtype=object),
            convergence_epoch=1,
            hist=np.array([{"train_total": [0.2], "valid_total": [0.1]}], dtype=object),
            crit_name=None,
            crit_epoch=np.array([]),
            crits=np.array([]),
        )
        (run_root / f"{run_name}_history.png").write_bytes(b"history")
        if config.get("plotting", {}).get("prediction", True):
            (run_root / f"{run_name}_prediction.png").write_bytes(b"prediction")
        if isinstance(config.get("cv"), dict):
            np.savez_compressed(
                run_root / f"{run_name}_cv.npz",
                all_results=np.array([], dtype=object),
                metric_name=config["cv"].get("metric", "total"),
                best_idx=0,
            )
            (run_root / "cv_results.png").write_bytes(b"cv")
        print("fake trainer finished", flush=True)


class _FailingTrainer(_FakeTrainer):
    def train(self):
        print("fake trainer failing", flush=True)
        raise RuntimeError("simulated training failure")


def bootstrap() -> None:
    trainer = (
        _FailingTrainer if os.environ.get("DYMAD_TRAINING_WORKER_MODE") == "fail" else _FakeTrainer
    )
    dymad.training.WeakFormTrainer = trainer
    dymad.training.NODETrainer = trainer
    dymad.training.LinearTrainer = trainer
    dymad.training.StackedTrainer = trainer
