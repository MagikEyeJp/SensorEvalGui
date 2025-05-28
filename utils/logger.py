import logging
from pathlib import Path
from typing import Optional

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure root logger with optional file output."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def log_memory_usage(prefix: str = "") -> None:
    """Log current process memory usage if psutil is available."""
    if not _HAS_PSUTIL:
        logging.debug("psutil not installed; cannot log memory usage")
        return
    try:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024**2
        logging.info("%sMemory usage: %.2f MB", prefix, mem_mb)
    except Exception as exc:
        logging.debug("Failed to log memory usage: %s", exc)
