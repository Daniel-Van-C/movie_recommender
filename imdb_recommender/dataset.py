from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

from .config import (
    BASICS_URL,
    RATINGS_URL,
    PRINCIPALS_URL,
    CREW_URL,
)
from .utils import _safe_int
from .downloader import DataDownloader

class IMDBDataset:
    """Load and prepare IMDb *movie* data and derived features."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.genre_matrix: Optional[np.ndarray] = None
        self.genre_labels: List[str] = []
        self.global_mean: float = 6.5
        self.cast_map: dict[str, set[str]] = {}
        self.crew_directors_map: dict[str, set[str]] = {}
        self.crew_writers_map: dict[str, set[str]] = {}

    # ---- Reading helpers -------------------------------------------------- #

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _read_tsv_gz_path(path: str) -> pd.DataFrame:
        """Read a gzipped TSV from a filesystem path (cached)."""
        return pd.read_csv(
            path,
            sep='\t',
            na_values=['\\N'],
            compression='gzip',
            engine='c',
            low_memory=False,
        )

    @staticmethod
    def _read_tsv_gz_any(path_or_buf) -> pd.DataFrame:
        """Read a gzipped TSV from a path or a file-like object."""
        if isinstance(path_or_buf, (str, Path)):
            return IMDBDataset._read_tsv_gz_path(str(path_or_buf))
        return pd.read_csv(
            path_or_buf,
            sep='\t',
            na_values=['\\N'],
            compression='gzip',
            engine='c',
            low_memory=False,
        )

    # ---- Hybrid helpers --------------------------------------------------- #

    @staticmethod
    def _parse_csv_nconst_field(val: str) -> set[str]:
        """Parse comma-separated nconsts (or '\\N') into a set."""
        if pd.isna(val) or val == "\\N" or not val:
            return set()
        return {x.strip() for x in str(val).split(',') if x and x != "\\N"}

    @staticmethod
    def _build_cast_map(principals: Optional[pd.DataFrame]) -> dict[str, set[str]]:
        if principals is None or principals.empty:
            return {}
        df = principals.loc[principals["nconst"].notna()].copy()
        if "category" in df.columns:
            df = df[df["category"].isin(["actor", "actress"])]
        grouped = df.groupby("tconst")["nconst"].apply(lambda s: set(s.astype(str).tolist()))
        return grouped.to_dict()

    @staticmethod
    def _build_crew_maps(crew: Optional[pd.DataFrame]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        if crew is None or crew.empty:
            return {}, {}
        cols = [c for c in ["tconst", "directors", "writers"] if c in crew.columns]
        df = crew[cols].copy()
        dir_map: dict[str, set[str]] = {}
        wri_map: dict[str, set[str]] = {}
        for _, row in df.iterrows():
            t = row["tconst"]
            dir_map[t] = IMDBDataset._parse_csv_nconst_field(row.get("directors", ""))
            wri_map[t] = IMDBDataset._parse_csv_nconst_field(row.get("writers", ""))
        return dir_map, wri_map

    # ---- Preprocessing ---------------------------------------------------- #

    @staticmethod
    def _preprocess(basics: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
        b = basics.copy()
        r = ratings.copy()

        b = b.loc[(b["titleType"] == "movie") & (b["isAdult"].fillna(0) == 0)]
        keep = ["tconst", "primaryTitle", "originalTitle", "startYear", "runtimeMinutes", "genres"]
        b = b[keep]

        b["startYear"] = b["startYear"].apply(_safe_int)
        b["runtimeMinutes"] = b["runtimeMinutes"].apply(_safe_int)

        b["genres"] = b["genres"].fillna("")
        b["genres_list"] = b["genres"].apply(lambda s: [] if not s or s == "\\N" else s.split(","))

        df = b.merge(r, on="tconst", how="left")
        df.rename(columns={"averageRating": "rating", "numVotes": "votes"}, inplace=True)

        def _disp(row: pd.Series) -> str:
            y = row["startYear"]
            return f"{row['primaryTitle']} ({int(y)})" if pd.notna(y) else str(row["primaryTitle"])

        df["display_title"] = df.apply(_disp, axis=1)
        df["rating"] = df["rating"].fillna(0.0)
        df["votes"] = df["votes"].fillna(0).astype(int)
        return df

    @staticmethod
    def _build_genre_matrix(genres_list: Sequence[Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
        mlb = MultiLabelBinarizer()
        G = mlb.fit_transform(genres_list)
        return G.astype(np.float32), list(mlb.classes_)

    # ---- Public load API -------------------------------------------------- #

    def load_from_uploads(self, basics_file, ratings_file, principals_file=None, crew_file=None) -> None:
        basics = self._read_tsv_gz_any(basics_file)
        ratings = self._read_tsv_gz_any(ratings_file)
        principals = self._read_tsv_gz_any(principals_file) if principals_file is not None else None
        crew = self._read_tsv_gz_any(crew_file) if crew_file is not None else None
        self._finalise(basics, ratings, principals, crew)

    def load_from_web(self, downloader: DataDownloader, include_cast_crew: bool = True) -> None:
        """Download IMDb datasets and compute features.

        If ``include_cast_crew`` is False, only basics+ratings are fetched.
        """
        to_get = [BASICS_URL, RATINGS_URL]
        if include_cast_crew:
            to_get += [PRINCIPALS_URL, CREW_URL]
        paths = downloader.download_many(to_get)
        basics = self._read_tsv_gz_path(str(paths[BASICS_URL]))
        ratings = self._read_tsv_gz_path(str(paths[RATINGS_URL]))
        principals = self._read_tsv_gz_path(str(paths[PRINCIPALS_URL])) if include_cast_crew else None
        crew = self._read_tsv_gz_path(str(paths[CREW_URL])) if include_cast_crew else None
        self._finalise(basics, ratings, principals, crew)

    def _finalise(self, basics: pd.DataFrame, ratings: pd.DataFrame, principals: Optional[pd.DataFrame] = None, crew: Optional[pd.DataFrame] = None) -> None:
        df = self._preprocess(basics, ratings)
        G, labels = self._build_genre_matrix(df["genres_list"])
        self.df = df
        self.genre_matrix = G
        self.genre_labels = labels
        self.global_mean = float(df.loc[df["rating"] > 0, "rating"].mean()) if (df["rating"] > 0).any() else 6.5
        self.cast_map = self._build_cast_map(principals)
        dmap, wmap = self._build_crew_maps(crew)
        self.crew_directors_map = dmap
        self.crew_writers_map = wmap
