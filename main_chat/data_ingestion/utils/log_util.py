"""
Centralized logging for the data sync pipeline.

Usage:
    from log_util import log, log_error, log_info, log_debug, set_verbosity, Verbosity

    set_verbosity(Verbosity.QUIET)  # Only errors
    set_verbosity(Verbosity.NORMAL)  # Start/end messages
    set_verbosity(Verbosity.VERBOSE)  # All output (default)
"""

import threading
from enum import IntEnum
from contextlib import contextmanager
from io import StringIO
import sys


class Verbosity(IntEnum):
    QUIET = 0  # Only errors and final results
    NORMAL = 1  # Start/finish messages and summaries
    VERBOSE = 2  # All messages (debug/progress)


_verbosity = Verbosity.VERBOSE
_lock = threading.Lock()


def set_verbosity(level: Verbosity | int) -> None:
    """Set the global verbosity level."""
    global _verbosity
    _verbosity = Verbosity(level)


def get_verbosity() -> Verbosity:
    """Get the current verbosity level."""
    return _verbosity


def log(message: str, level: Verbosity = Verbosity.NORMAL, **kwargs) -> None:
    """Print message if current verbosity >= level. Supports print() kwargs like end, flush."""
    if _verbosity >= level:
        with _lock:
            print(message, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Always log errors (level QUIET)."""
    log(f"✗ {message}", Verbosity.QUIET, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Log warnings at NORMAL level."""
    log(f"⚠ {message}", Verbosity.NORMAL, **kwargs)


def log_success(message: str, **kwargs) -> None:
    """Log success messages at NORMAL level."""
    log(f"✔ {message}", Verbosity.NORMAL, **kwargs)


def log_info(message: str, **kwargs) -> None:
    """Log info messages at NORMAL level."""
    log(message, Verbosity.NORMAL, **kwargs)


def log_debug(message: str, **kwargs) -> None:
    """Log debug/progress messages at VERBOSE level."""
    log(message, Verbosity.VERBOSE, **kwargs)


def log_progress(message: str, **kwargs) -> None:
    """Log progress messages at VERBOSE level (alias for log_debug)."""
    log(message, Verbosity.VERBOSE, **kwargs)


@contextmanager
def capture_output():
    """
    Context manager that captures stdout.
    Useful for capturing verbose output to replay on error.

    Usage:
        with capture_output() as captured:
            do_something_verbose()
        if error_occurred:
            print(captured.getvalue())
    """
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        yield captured
    finally:
        sys.stdout = old_stdout


@contextmanager
def suppress_unless_error():
    """
    Context manager that suppresses stdout unless an exception occurs.
    If an exception is raised, the captured output is printed before re-raising.

    Usage:
        with suppress_unless_error():
            do_something_verbose()  # Only shown if exception raised
    """
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        yield
    except Exception:
        old_stdout.write(captured.getvalue())
        raise
    finally:
        sys.stdout = old_stdout
