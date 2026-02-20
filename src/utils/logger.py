"""Structured logging configuration using structlog.

JSON output in production, colored console in development.
Secret masking processor ensures passwords/tokens never leak into logs.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import structlog

# Patterns that indicate a secret value
_SECRET_PATTERNS = re.compile(
    r"(password|token|secret|api_key|api[-_]?key|session[-_]?token|authorization)",
    re.IGNORECASE,
)
_MASK = "***REDACTED***"


def _mask_secrets(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Mask any key/value pairs that look like secrets."""
    for key in list(event_dict.keys()):
        if _SECRET_PATTERNS.search(key):
            event_dict[key] = _MASK
        elif isinstance(event_dict[key], str):
            val = event_dict[key]
            # Mask values that look like known token formats (any length)
            if val.startswith(("xoxb-", "xapp-", "sk-", "sk-ant-")):
                event_dict[key] = val[:8] + "..." + _MASK
    return event_dict


def setup_logging(log_level: str = "INFO", json_output: bool | None = None) -> None:
    """Configure structlog and stdlib logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: Force JSON output. If None, auto-detect from MODE env var.
    """
    if json_output is None:
        json_output = os.getenv("MODE", "paper") != "dev"

    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _mask_secrets,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("aiohttp", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(module: str) -> structlog.stdlib.BoundLogger:
    """Get a bound logger with module context.

    Args:
        module: Module name for context binding.

    Returns:
        A structlog bound logger instance.
    """
    return structlog.get_logger(module=module)
