import logging
from datetime import datetime
from pathlib import Path


def setup_file_logger() -> tuple[logging.Logger, Path]:
    logs_dir = Path("runs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    return logger, log_path
