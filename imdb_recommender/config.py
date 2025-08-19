from __future__ import annotations

from pathlib import Path

# URLs for IMDb datasets
BASICS_URL: str = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL: str = "https://datasets.imdbws.com/title.ratings.tsv.gz"
PRINCIPALS_URL: str = "https://datasets.imdbws.com/title.principals.tsv.gz"
CREW_URL: str = "https://datasets.imdbws.com/title.crew.tsv.gz"

# Download/io settings
DL_CHUNK_BYTES: int = 4_194_304  # 4 MiB chunks
CACHE_DIR: Path = Path.home() / ".cache" / "imdb_datasets"

# Recommender defaults
DEFAULT_MIN_VOTES: int = 500
DEFAULT_SHOW: int = 25
MAX_SHOW: int = 100
