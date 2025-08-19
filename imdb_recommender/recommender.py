from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_MIN_VOTES, DEFAULT_SHOW

@dataclass(frozen=True)
class Weights:
    """Relative weights for combining similarity and rating components."""
    genre: float = 0.5
    rating: float = 0.2
    cast: float = 0.2
    crew: float = 0.1

    def normalised(self) -> "Weights":
        s = self.genre + self.rating + self.cast + self.crew
        if s <= 0:
            return Weights(1.0, 0.0, 0.0, 0.0)
        return Weights(self.genre / s, self.rating / s, self.cast / s, self.crew / s)

class Recommender:
    """Content-based recommender with genre cosine, Bayesian rating, cast overlap, crew overlap."""

    def __init__(self, dataset, weights: Weights, min_votes: int = DEFAULT_MIN_VOTES) -> None:
        if dataset.df is None or dataset.genre_matrix is None:
            raise ValueError("Dataset not loaded.")
        self.ds = dataset
        self.w = weights.normalised()
        self.min_votes = int(min_votes)

    @staticmethod
    def _bayesian_weighted_rating(r: np.ndarray, v: np.ndarray, C: float, m: int) -> np.ndarray:
        return (v / (v + m)) * r + (m / (v + m)) * C

    def _genre_similarity(self, fav_idx: List[int]) -> np.ndarray:
        G = self.ds.genre_matrix
        centroid = G[fav_idx].mean(axis=0, keepdims=True)
        return cosine_similarity(G, centroid).ravel().astype(np.float32)

    def _rating_component(self) -> np.ndarray:
        df = self.ds.df
        m_threshold = int(np.percentile(df["votes"], 75))
        r = df["rating"].to_numpy(np.float32)
        v = df["votes"].to_numpy(np.float32)
        bw = self._bayesian_weighted_rating(r, v, C=self.ds.global_mean, m=m_threshold)
        mn, mx = float(bw.min()), float(bw.max())
        return (bw - mn) / (mx - mn) if mx > mn else np.zeros_like(bw, dtype=np.float32)

    def recommend(self, fav_tconsts: Iterable[str], k: int = DEFAULT_SHOW, candidate_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
        df = self.ds.df
        fav_idx = df.index[df["tconst"].isin(list(fav_tconsts))].to_list()
        if not fav_idx:
            return df.head(0).copy()

        sim = self._genre_similarity(fav_idx)
        rating = self._rating_component()

        cast_map = self.ds.cast_map
        dir_map = self.ds.crew_directors_map
        wri_map = self.ds.crew_writers_map
        fav_tconsts = list(fav_tconsts)

        fav_cast_union: set[str] = set().union(*[cast_map.get(t, set()) for t in fav_tconsts]) if cast_map else set()
        fav_dw_union: set[str] = set().union(
            *[dir_map.get(t, set()) | wri_map.get(t, set()) for t in fav_tconsts]
        ) if (dir_map or wri_map) else set()

        def jaccard(a: set[str], b: set[str]) -> float:
            if not a and not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        is_fav = df["tconst"].isin(list(fav_tconsts)).to_numpy()
        passes_votes = (df["votes"].to_numpy() >= self.min_votes)
        valid_mask = (~is_fav) & passes_votes
        if candidate_mask is not None:
            valid_mask &= candidate_mask
        if not bool(valid_mask.any()):
            return df.head(0).copy()

        valid_idx = np.where(valid_mask)[0]
        cast_scores = np.zeros(len(df), dtype=np.float32)
        crew_scores = np.zeros(len(df), dtype=np.float32)
        if fav_cast_union and cast_map:
            for i in valid_idx:
                t = df.iat[i, df.columns.get_loc("tconst")]
                cast_scores[i] = jaccard(cast_map.get(t, set()), fav_cast_union)
        if fav_dw_union and (dir_map or wri_map):
            for i in valid_idx:
                t = df.iat[i, df.columns.get_loc("tconst")]
                crew_set = dir_map.get(t, set()) | wri_map.get(t, set())
                crew_scores[i] = jaccard(crew_set, fav_dw_union)

        score_all = (
            self.w.genre * sim +
            self.w.rating * rating +
            self.w.cast * cast_scores +
            self.w.crew * crew_scores
        )

        top_k = min(int(k), len(valid_idx))
        if top_k <= 0:
            return df.head(0).copy()

        pool = min(len(valid_idx), max(top_k * 5, top_k + 5))
        v_scores = score_all[valid_idx]
        pool_idx_local = np.argpartition(v_scores, -pool)[-pool:]
        top_local = pool_idx_local[np.argsort(v_scores[pool_idx_local])[::-1]][:top_k]
        idx = valid_idx[top_local]

        recs = df.iloc[idx][["tconst", "display_title", "genres", "startYear", "rating", "votes"]].copy()
        recs["score"] = score_all[idx]
        return recs
