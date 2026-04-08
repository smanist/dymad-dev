import copy
import logging
import os
import sys
from datetime import datetime
from typing import Any

import yaml


def setup_logging(config_path: str = "", mode: str = "info", prefix=".") -> None:
    """
    Setup logging configuration based on the config file.
    Assuming the config file name is in the format '<case>.yaml'

    Args:
        config_path (str): Path to the configuration file.
        mode (str): Logging mode, either 'debug' or 'info'. Default is 'info'.
        prefix (str): Directory prefix for the log file. Default is '.' (current directory).
    """
    _l = logging.DEBUG if mode == "debug" else logging.INFO
    if config_path == "":
        # If no config path is provided, log to stdout
        logging.basicConfig(
            stream=sys.stdout,
            level=_l,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        return
    _t = str(datetime.now())
    _t = _t.split(".")[0].replace(" ", "-").replace(":", "-")
    if prefix != ".":
        os.makedirs(prefix, exist_ok=True)
    logging.basicConfig(
        filename=f"{prefix}/{config_path.split('.')[0]}_{_t}.log",
        filemode="w",
        level=_l,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Having force=True flushes and closes any existing handlers,
    # so no need to close them manually here.
    return


def config_logger(logger: logging.Logger, mode: str = "info", prefix=".") -> None:
    """
    Configure a logger with specified mode and prefix.  The function is intended for
    a logger instantiated in a class.

    Args:
        logger (logging.Logger): Logger instance to configure.
        mode (str): Logging mode, either 'debug' or 'info'. Default is 'info'.
        prefix (str): Prefix for the log file. Default is '.' (current directory).
    """
    _l = logging.DEBUG if mode == "debug" else logging.INFO
    logger.setLevel(_l)

    if logger.handlers:
        # Avoid duplicate handlers if called twice
        return

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    if prefix == "":
        # If no config path is provided, log to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_l)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logging.getLogger().handlers = []
        return

    # Otherwise we set up the logging to file
    _t = str(datetime.now())
    _t = _t.split(".")[0].replace(" ", "-").replace(":", "-")
    handler = logging.FileHandler(f"{prefix}_{_t}.log")
    handler.setLevel(_l)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logging.getLogger().handlers = []


def load_config(config_path: str, config_mod: dict[str, Any] | None = None) -> dict:
    """
    Load a YAML configuration file and optionally merge with a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.
        config_mod (dict, optional): Dictionary to merge into the config.

    Returns:
        dict: Merged configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_mod is not None:
        if not isinstance(config_mod, dict):
            raise TypeError("config_mod must be a dictionary.")
        for key, value in config_mod.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    _normalize_legacy_training_config(config)
    return config


def _is_optimizer_phase_entry(entry: object) -> bool:
    return isinstance(entry, dict) and (entry.get("type") == "optimizer" or "trainer" in entry)


def _normalized_legacy_training_phase(training_cfg: dict, *, name: str, trainer: str) -> dict:
    phase = copy.deepcopy(training_cfg)
    phase.setdefault("type", "optimizer")
    phase.setdefault("name", name)
    phase.setdefault("trainer", trainer)
    return phase


def _first_optimizer_phase_index(phases: object) -> int:
    if not isinstance(phases, list):
        return 0
    for index, phase in enumerate(phases):
        if _is_optimizer_phase_entry(phase):
            return index
    return 0


def _rewrite_legacy_training_param_grid(config: dict, phase_index: int) -> None:
    cv_cfg = config.get("cv")
    if not isinstance(cv_cfg, dict):
        return
    param_grid = cv_cfg.get("param_grid")
    if not isinstance(param_grid, dict):
        return
    rewritten = {}
    for key, value in param_grid.items():
        if isinstance(key, str) and key.startswith("training."):
            suffix = key.split(".", 1)[1]
            rewritten[f"phases.{phase_index}.{suffix}"] = value
        else:
            rewritten[key] = value
    cv_cfg["param_grid"] = rewritten


def _normalize_legacy_training_config(config: dict) -> None:
    phases = config.get("phases")
    training_cfg = config.get("training")
    if isinstance(training_cfg, dict):
        if isinstance(phases, list):
            merged = False
            for index, phase in enumerate(phases):
                if not _is_optimizer_phase_entry(phase):
                    continue
                merged_phase = copy.deepcopy(phase)
                merged_phase.update(copy.deepcopy(training_cfg))
                merged_phase.setdefault("type", "optimizer")
                phases[index] = merged_phase
                _rewrite_legacy_training_param_grid(config, index)
                merged = True
                break
            if not merged:
                phases.insert(
                    0,
                    _normalized_legacy_training_phase(
                        training_cfg,
                        name="phase_0",
                        trainer="NODE",
                    ),
                )
                _rewrite_legacy_training_param_grid(config, 0)
        else:
            config["phases"] = [
                _normalized_legacy_training_phase(training_cfg, name="phase_0", trainer="NODE")
            ]
            _rewrite_legacy_training_param_grid(config, 0)
        del config["training"]
    else:
        _rewrite_legacy_training_param_grid(config, _first_optimizer_phase_index(phases))
