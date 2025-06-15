import logging
import os


class RankFormatter(logging.Formatter):
    """Custom formatter that includes rank information from environment variable."""

    def __init__(self, fmt=None, datefmt="%Y-%m-%d %H:%M:%S"):
        # Base format without rank
        self.base_fmt = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
        # Format with rank
        self.rank_fmt = "[rank=%(rank)s][%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
        super().__init__(fmt, datefmt)

    def format(self, record):
        rank = os.getenv("ENV_RANK")

        if rank is not None:
            # Use rank format and add rank to record
            record.rank = rank
            self._style._fmt = self.rank_fmt
        else:
            # Use base format without rank
            self._style._fmt = self.base_fmt

        return super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with rank information.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create handler and formatter
    handler = logging.StreamHandler()
    formatter = RankFormatter()
    handler.setFormatter(formatter)

    # Configure logger
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs

    return logger
