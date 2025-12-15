# whisperjav/data/hallucination_filters/__init__.py
"""
Bundled hallucination filter data.

These files are snapshots of the online hallucination filter lists,
bundled with the package for:
1. Offline use (China, air-gapped networks)
2. Faster startup (no network request needed)
3. Fallback when online sources are unavailable

Files:
- filter_list_v08.json: Exact match phrases by language
- regexp_v09.json: Regex patterns for hallucination detection

To update bundled files, run:
    python -m whisperjav.utils.update_bundled_data
"""
from pathlib import Path

# Get the directory containing the bundled data files
BUNDLED_DATA_DIR = Path(__file__).parent


def get_bundled_filter_list_path() -> Path:
    """Return path to bundled filter list JSON."""
    return BUNDLED_DATA_DIR / "filter_list_v08.json"


def get_bundled_regexp_path() -> Path:
    """Return path to bundled regexp patterns JSON."""
    return BUNDLED_DATA_DIR / "regexp_v09.json"
