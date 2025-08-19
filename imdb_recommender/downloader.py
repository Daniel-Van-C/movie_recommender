from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Sequence, Optional
import hashlib

import requests
import streamlit as st

from .config import DL_CHUNK_BYTES, CACHE_DIR

def _filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1] or hashlib.sha1(url.encode()).hexdigest() + ".gz"

class DataDownloader:
    """Downloader with simple disk cache and parallel fetches."""

    def __init__(self, chunk_size: int = DL_CHUNK_BYTES, cache_dir: Optional[Path] = None) -> None:
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _head_len(self, url: str) -> Optional[int]:
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            if r.ok:
                cl = r.headers.get("content-length")
                return int(cl) if cl is not None else None
        except requests.RequestException:
            return None
        return None

    def _download_stream(self, url: str, dest: Path) -> Path:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=self.chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        if total and written < total:
            raise IOError(f"Incomplete download for {url}: got {written} of {total} bytes")
        return dest

    def download_if_needed(self, url: str) -> Path:
        """Return cached file if present; otherwise download and cache it."""
        target = self.cache_dir / _filename_from_url(url)
        if target.exists() and target.stat().st_size > 0:
            exp = self._head_len(url)
            if exp is None or target.stat().st_size == exp:
                return target
        tmp = target.with_suffix(target.suffix + ".part")
        p = self._download_stream(url, tmp)
        p.replace(target)
        return target

    def download_many(self, urls: Sequence[str]) -> Dict[str, Path]:
        """Fetch several URLs in parallel; return mapping url->cached path."""
        progress = st.progress(0.0)
        results: Dict[str, Path] = {}
        total = len(urls) or 1
        completed = 0
        with ThreadPoolExecutor(max_workers=min(4, total)) as ex:
            for url, path in ex.map(lambda u: (u, self.download_if_needed(u)), urls):
                results[url] = path
                completed += 1
                progress.progress(completed / total)
        progress.empty()
        return results
