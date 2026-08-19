import logging
from logging.handlers import RotatingFileHandler

from config import LOG_FILE, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Parameters
    ----------
    name : str
        Name of the module requesting the logger.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -------------------------
    # Console Handler
    # -------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # -------------------------
    # File Handler
    # -------------------------

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger