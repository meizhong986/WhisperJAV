"""
Pytest configuration for WhisperJAV tests.

Registers custom markers used across test suites.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (longer than 60 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
