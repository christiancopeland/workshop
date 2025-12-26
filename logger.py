"""
Workshop Logger
Centralized logging with file output for debugging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "workshop",
    log_dir: Path = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Console shows warnings and errors only.
    File captures everything for debugging.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent / "data" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - captures everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"workshop_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - warnings and errors only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log startup
    logger.info("=" * 60)
    logger.info(f"Workshop session started at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    
    # Store log file path for reference
    logger.log_file = log_file
    
    return logger


# Create default logger instance
log = setup_logger()


def get_logger(name: str = None) -> logging.Logger:
    """Get a child logger with the given name"""
    if name:
        return logging.getLogger(f"workshop.{name}")
    return log
