# app/logger_config.py
import logging
import os
from logging.handlers import RotatingFileHandler
import yaml
 
def setup_logger(name: str, cfg: dict) -> logging.Logger:
    logger = logging.getLogger(name)

    # Clear existing handlers completely before re-adding
    # This prevents duplicate log entries on Streamlit reruns
    if logger.handlers:
        logger.handlers.clear()

    level_str = cfg.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)
 
    if logger.handlers:
        return logger  # avoid duplicate handlers on reload
 
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
 
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
 
    # Rotating file handler
    log_path = os.path.join(cfg['log_dir'], f'{name}.log')
    os.makedirs(cfg['log_dir'], exist_ok=True)
    fh = RotatingFileHandler(
        log_path,
        maxBytes=cfg.get('max_bytes', 10_485_760),
        backupCount=cfg.get('backup_count', 5),
        encoding='utf-8'
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
 
    return logger
