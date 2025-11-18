import logging
from colorlog import ColoredFormatter


def create_logger(name: str = "logger", level: str = "INFO", trace: bool = False):
    formatter = ColoredFormatter(
        (
            "%(yellow)s%(asctime)s%(reset)s"
            + " | %(log_color)s%(levelname)s%(reset)s"
            + (" | %(purple)s%(filename)s:LINE%(lineno)d%(reset)s" if trace else "")
            + " | %(blue)s%(message)s%(reset)s"
        ),
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
