#!/usr/bin/env python3
"""Simple logging setup for WhisperJAV."""
"""UTF-8-safe logging setup for WhisperJAV."""

import logging
import sys
import os
import io
from pathlib import Path
from typing import Optional


class UTF8StreamHandler(logging.StreamHandler):
    """Custom stream handler that forces UTF-8 encoding for console output."""
    def __init__(self, stream=None):
        if stream is None:
            stream = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        super().__init__(stream)


def setup_logger(name: str = "whisperjav", 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:
    """Setup a UTF-8-safe logger with console and optional file output."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = UTF8StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8', errors='replace')
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global logger instance (can be reused across your app)
logger = setup_logger()
