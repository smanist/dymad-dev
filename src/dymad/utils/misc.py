from datetime import datetime
import logging
import os
import yaml

def setup_logging(config_path: str, mode: str = 'info', prefix='.') -> None:
    """
    Setup logging configuration based on the config file.
    Assuming the config file name is in the format '<case>.yaml'

    Args:
        config_path (str): Path to the configuration file.
        mode (str): Logging mode, either 'debug' or 'info'. Default is 'info'.
        prefix (str): Directory prefix for the log file. Default is '.' (current directory).
    """
    _l = logging.DEBUG if mode == 'debug' else logging.INFO
    _t = str(datetime.now())
    _t = _t.split('.')[0].replace(' ', '-').replace(':', '-')
    if prefix != '.':
        os.makedirs(prefix, exist_ok=True)
    logging.basicConfig(
        filename=f'{prefix}/{config_path.split(".")[0]}_{_t}.log',  
        filemode='w',  
        level=_l,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    # Having force=True flushes and closes any existing handlers,
    # so no need to close them manually here.

def load_config(config_path: str, config_mod: dict = None) -> dict:
    """
    Load a YAML configuration file and optionally merge with a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.
        config_mod (dict, optional): Dictionary to merge into the config.

    Returns:
        dict: Merged configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_mod is not None:
        if not isinstance(config_mod, dict):
            raise TypeError("config_mod must be a dictionary.")
        config.update(config_mod)
    return config