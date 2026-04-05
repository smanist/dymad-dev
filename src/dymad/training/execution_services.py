from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import torch

from dymad.utils import config_logger


@dataclass(frozen=True)
class ExecutionServices:
    """Non-checkpointable runtime policy for one training execution path."""

    device: torch.device
    checkpoint_prefix: str
    results_prefix: str
    log_level: str = "info"
    log_stdout: bool = False

    @classmethod
    def _select_device(
        cls,
        default_device: Optional[torch.device] = None,
    ) -> torch.device:
        if default_device is not None:
            return default_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any] | None,
        default_device: Optional[torch.device] = None,
    ) -> "ExecutionServices":
        cfg = config or {}
        path_cfg = cfg.get("path", {})
        checkpoint_prefix = path_cfg.get("checkpoint_prefix", ".")
        results_prefix = path_cfg.get("results_prefix", checkpoint_prefix)
        log_cfg = cfg.get("log", {})
        return cls(
            device=cls._select_device(default_device=default_device),
            checkpoint_prefix=checkpoint_prefix,
            results_prefix=results_prefix,
            log_level=log_cfg.get("level", "info"),
            log_stdout=log_cfg.get("stdout", False),
        )

    @classmethod
    def from_driver_config(
        cls,
        base_config: Dict[str, Any],
        config_path: str,
        default_device: Optional[torch.device] = None,
    ) -> "ExecutionServices":
        root = os.path.dirname(config_path) or "."
        base_name = base_config["model"]["name"]
        prefix = os.path.join(root, base_name)
        log_cfg = base_config.get("log", {})
        return cls(
            device=cls._select_device(default_device=default_device),
            checkpoint_prefix=prefix,
            results_prefix=prefix,
            log_level=log_cfg.get("level", "info"),
            log_stdout=log_cfg.get("stdout", False),
        )

    def with_paths(
        self,
        *,
        checkpoint_prefix: Optional[str] = None,
        results_prefix: Optional[str] = None,
    ) -> "ExecutionServices":
        return replace(
            self,
            checkpoint_prefix=self.checkpoint_prefix if checkpoint_prefix is None else checkpoint_prefix,
            results_prefix=self.results_prefix if results_prefix is None else results_prefix,
        )

    def with_device(self, device: torch.device) -> "ExecutionServices":
        return replace(self, device=device)

    def apply_to_config(self, config: Dict[str, Any] | None) -> Dict[str, Any]:
        cfg = copy.deepcopy(config or {})
        cfg.setdefault("path", {})
        cfg["path"]["checkpoint_prefix"] = self.checkpoint_prefix
        cfg["path"]["results_prefix"] = self.results_prefix
        cfg.setdefault("log", {})
        cfg["log"]["level"] = self.log_level
        cfg["log"]["stdout"] = self.log_stdout
        return cfg

    def ensure_artifact_dirs(self) -> None:
        os.makedirs(self.checkpoint_prefix, exist_ok=True)
        os.makedirs(self.results_prefix, exist_ok=True)

    def checkpoint_file(self, file_name: str) -> str:
        return os.path.join(self.checkpoint_prefix, file_name)

    def logger_prefix(self, default_name: str) -> str:
        if self.log_stdout:
            return ""
        return os.path.join(self.results_prefix, default_name)

    def configure_logger(self, name: str, prefix: Optional[str] = None) -> logging.Logger:
        logger = logging.getLogger(name)
        config_logger(
            logger,
            mode=self.log_level,
            prefix=self.logger_prefix(name) if prefix is None else prefix,
        )
        return logger
