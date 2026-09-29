"""Logging setup for the framework."""
import logging
import sys


def get_logger(name="terraforge", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
