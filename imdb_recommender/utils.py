from __future__ import annotations

from typing import Optional

def _safe_int(x) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    try:
        return int(x)
    except Exception:
        return None
