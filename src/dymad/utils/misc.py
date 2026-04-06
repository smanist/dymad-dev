import logging
import os
import sys
from datetime import datetime

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


def load_config(config_path: str, config_mod: dict = None) -> dict:
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
    return config
