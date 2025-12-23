"""
Logging Configuration for Sharp Sports Predictor.

Provides structured logging with:
- File and console handlers
- Log rotation
- Context-aware logging
- Performance timing
"""

import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional


# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory
LOG_DIR = Path(__file__).parent.parent.parent / "logs"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file name
        enable_console: Whether to log to console
        enable_file: Whether to log to file

    Returns:
        Root logger instance
    """
    # Create logs directory if needed
    if enable_file:
        LOG_DIR.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger("sharp_sports")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file:
        if log_file is None:
            log_file = f"sharp_sports_{datetime.now().strftime('%Y%m%d')}.log"

        file_path = LOG_DIR / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__ of the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"sharp_sports.{name}")


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, logger: logging.Logger, context: dict):
        self.logger = logger
        self.context = context
        self._original_extra = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Error in context {self.context}: {exc_val}",
                exc_info=True,
            )
        return False


@contextmanager
def log_timing(logger: logging.Logger, operation: str):
    """
    Context manager to log the duration of an operation.

    Usage:
        with log_timing(logger, "model training"):
            train_model()
    """
    start_time = time.perf_counter()
    logger.info(f"Starting: {operation}")
    try:
        yield
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Failed: {operation} after {elapsed:.2f}s - {e}")
        raise
    else:
        elapsed = time.perf_counter() - start_time
        logger.info(f"Completed: {operation} in {elapsed:.2f}s")


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with arguments and return values.

    Usage:
        @log_function_call()
        def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} returned successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} raised {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


class StructuredLogger:
    """
    Structured logger for key events with consistent formatting.

    Provides methods for common logging patterns in the betting system.
    """

    def __init__(self, name: str):
        self.logger = get_logger(name)

    def log_prediction(
        self,
        game: str,
        predicted_margin: float,
        vegas_line: float,
        edge: float,
        confidence: float,
    ):
        """Log a prediction event."""
        self.logger.info(
            f"PREDICTION | Game: {game} | "
            f"Predicted: {predicted_margin:+.1f} | "
            f"Vegas: {vegas_line:+.1f} | "
            f"Edge: {edge:+.1f} | "
            f"Confidence: {confidence:.1%}"
        )

    def log_bet_recommendation(
        self,
        game: str,
        side: str,
        bet_size: float,
        kelly_pct: float,
        win_prob: float,
    ):
        """Log a bet recommendation."""
        self.logger.info(
            f"BET | Game: {game} | "
            f"Side: {side} | "
            f"Size: ${bet_size:.2f} | "
            f"Kelly: {kelly_pct:.2%} | "
            f"Win Prob: {win_prob:.1%}"
        )

    def log_model_performance(
        self,
        model_name: str,
        mae: float,
        samples: int,
        improvement: Optional[float] = None,
    ):
        """Log model performance metrics."""
        msg = (
            f"MODEL | {model_name} | "
            f"MAE: {mae:.2f} pts | "
            f"Samples: {samples}"
        )
        if improvement is not None:
            msg += f" | Improvement: {improvement:+.2f} pts"
        self.logger.info(msg)

    def log_api_call(
        self,
        endpoint: str,
        params: dict,
        success: bool,
        response_size: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """Log an API call."""
        if success:
            self.logger.debug(
                f"API | {endpoint} | Params: {params} | "
                f"Size: {response_size or 'N/A'}"
            )
        else:
            self.logger.warning(
                f"API ERROR | {endpoint} | Params: {params} | "
                f"Error: {error}"
            )

    def log_data_validation(
        self,
        data_type: str,
        total_records: int,
        valid_records: int,
        issues: Optional[list] = None,
    ):
        """Log data validation results."""
        valid_pct = (valid_records / total_records * 100) if total_records > 0 else 0
        self.logger.info(
            f"VALIDATION | {data_type} | "
            f"Total: {total_records} | "
            f"Valid: {valid_records} ({valid_pct:.1f}%)"
        )
        if issues:
            for issue in issues[:5]:  # Log first 5 issues
                self.logger.warning(f"VALIDATION ISSUE | {issue}")
