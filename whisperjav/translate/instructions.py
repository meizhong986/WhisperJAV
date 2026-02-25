"""
Instructions management - fetch from Gist, cache, and fallback.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Default instruction file URLs (Gist raw URLs)
DEFAULT_INSTRUCTION_URLS = {
    'standard': 'https://gist.githubusercontent.com/meizhong986/fd38f54ccbf6c1df3f04019a4875f6db/raw/translationJAV-standard.txt',
    'pornify': 'https://gist.githubusercontent.com/meizhong986/bd34016a3d84f16f1086e8caf18e0d0f/raw/translationJAV-pornify.txt'
}


def get_cache_dir() -> Path:
    """Get platform-specific cache directory."""
    from .settings import get_settings_path
    return get_settings_path().parent / 'cache'


def get_cache_path(tone: str) -> Path:
    """Get cache file path for a specific tone."""
    cache_dir = get_cache_dir()
    return cache_dir / f'instruction_{tone}.txt'


def get_cache_index_path() -> Path:
    """Get cache index file path."""
    return get_cache_dir() / 'cache_index.json'


def load_cache_index() -> dict:
    """Load cache index containing ETags and metadata."""
    index_path = get_cache_index_path()

    if not index_path.exists():
        return {}

    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache index: {e}")
        return {}


def save_cache_index(index: dict):
    """Save cache index."""
    index_path = get_cache_index_path()

    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache index: {e}")


def fetch_from_gist(url: str, etag: Optional[str] = None, timeout: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch instruction content from Gist URL.

    Args:
        url: Gist raw URL
        etag: Previous ETag for conditional requests
        timeout: Request timeout in seconds

    Returns:
        Tuple of (content, new_etag)
        - (None, etag) if not modified
        - (content, new_etag) if updated
        - (None, None) on error
    """
    headers = {}
    if etag:
        headers['If-None-Match'] = etag

    try:
        logger.debug(f"Fetching instructions from {url}")
        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 304:
            # Not modified
            logger.debug("Instructions unchanged (304)")
            return None, etag

        if response.status_code == 200:
            content = response.text
            new_etag = response.headers.get('ETag')
            logger.debug(f"Fetched instructions, ETag: {new_etag}")
            return content, new_etag

        logger.warning(f"Unexpected status code: {response.status_code}")
        return None, None

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch instructions from {url}: {e}")
        return None, None


def load_from_cache(tone: str) -> Optional[str]:
    """Load instructions from cache."""
    cache_path = get_cache_path(tone)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Loaded instructions from cache: {cache_path}")
        return content
    except Exception as e:
        logger.warning(f"Failed to load from cache: {e}")
        return None


def save_to_cache(tone: str, content: str, etag: Optional[str], url: str):
    """Save instructions to cache with metadata."""
    cache_path = get_cache_path(tone)

    try:
        # Save content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Update index
        index = load_cache_index()
        index[tone] = {
            'url': url,
            'etag': etag,
            'timestamp': time.time()
        }
        save_cache_index(index)

        logger.debug(f"Saved instructions to cache: {cache_path}")

    except Exception as e:
        logger.warning(f"Failed to save to cache: {e}")


def load_bundled_default(tone: str) -> Optional[str]:
    """Load bundled default instruction file."""
    try:
        # Look for bundled defaults in package
        from importlib import resources
        try:
            # Python 3.9+
            files = resources.files('whisperjav.translate.defaults')
            default_file = files / f'{tone}.txt'
            if default_file.is_file():
                return default_file.read_text(encoding='utf-8')
        except AttributeError:
            # Fallback for older Python
            with resources.open_text('whisperjav.translate.defaults', f'{tone}.txt') as f:
                return f.read()
    except Exception as e:
        logger.debug(f"No bundled default for tone '{tone}': {e}")
        return None


def get_instruction_content(tone: str = 'standard', refresh: bool = False) -> Optional[str]:
    """
    Get instruction content with fallback strategy.

    Strategy:
    1. Fetch from Gist (with ETag caching)
    2. Load from cache (if fetch fails or not modified)
    3. Load bundled default (if all else fails)

    Args:
        tone: Instruction tone (standard, pornify, etc.)
        refresh: Force refresh from network

    Returns:
        Instruction content or None
    """
    # Get URL for this tone
    url = DEFAULT_INSTRUCTION_URLS.get(tone)

    if not url:
        logger.warning(f"No URL configured for tone: {tone}")
        return load_bundled_default(tone)

    # Try fetching (with caching)
    if not refresh:
        # Load cache index for ETag
        index = load_cache_index()
        cached_etag = None
        if tone in index:
            cached_etag = index[tone].get('etag')

        # Fetch with conditional request
        content, new_etag = fetch_from_gist(url, etag=cached_etag)

        # If not modified, load from cache
        if content is None and new_etag is not None:
            cached_content = load_from_cache(tone)
            if cached_content:
                return cached_content

        # If fetched, save to cache
        if content is not None:
            save_to_cache(tone, content, new_etag, url)
            return content

    else:
        # Force refresh
        logger.info("Forcing instruction refresh...")
        content, new_etag = fetch_from_gist(url)
        if content is not None:
            save_to_cache(tone, content, new_etag, url)
            return content

    # Fallback to cache
    cached_content = load_from_cache(tone)
    if cached_content:
        logger.info("Using cached instructions (fetch failed)")
        return cached_content

    # Final fallback to bundled default
    bundled = load_bundled_default(tone)
    if bundled:
        logger.info("Using bundled default instructions")
        return bundled

    logger.error(f"Failed to load instructions for tone: {tone}")
    return None
