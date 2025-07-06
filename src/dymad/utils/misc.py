from datetime import datetime
import logging

def setup_logging(config_path: str, mode: str = 'info') -> None:
    """
    Setup logging configuration based on the config file.
    Assuming the config file name is in the format '<case>.yaml'
    """
    _l = logging.DEBUG if mode == 'debug' else logging.INFO
    _t = str(datetime.now())
    _t = _t.split('.')[0].replace(' ', '-')
    logging.basicConfig(
        filename=f'{config_path.split(".")[0]}_{_t}.log',  
        filemode='w',  
        level=_l,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

def close_logging() -> None:
    logger = logging.getLogger(__name__)
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()