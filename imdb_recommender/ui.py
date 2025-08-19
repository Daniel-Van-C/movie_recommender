from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.manifold import TSNE

from .config import DEFAULT_MIN_VOTES, DEFAULT_SHOW, MAX_SHOW
from .downloader import DataDownloader
from .dataset import IMDBDataset
from .recommender import Weights, Recommender

@dataclass
class AppState:
    favs: set[str]
    search: str

    @classmethod
    def new(cls) -> "AppState":
        return cls(favs=set(), search="")

def main() -> None:
    st.set_page_config(page_title="IMDb Recommender (OOP)", layout="wide")
    st.markdown(
        """
        <style>
        :root { --accent: #2563eb; }
        h1, h2, h3 { color: var(--accent); }
        .stButton>button { background: var(--accent); color: white; border: 0; border-radius: 8px; }
        [data-testid="stSidebar"] > div:first-child { background: #f8fafc; }
        .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🎬 IMDb Favourites-Based Movie Recommender")

    with st.expander("How recommendations are computed", expanded=False):
        st.markdown(
            """
            **Matching logic**

            Your favourites form a *profile*. We score every candidate movie with:
            - **Genre similarity**: cosine similarity between your favourites' average genre vector and each movie's genre vector.
            - **Quality prior**: Bayesian-weighted IMDb rating (accounts for vote counts).
            - **Cast overlap** *(if principals/crew data available)*: Jaccard overlap between your favourites' cast and each movie's cast.
            - **Director/Writer overlap** *(if crew data available)*: Jaccard overlap of directors/writers.
            """
        )

    st.info("To use this app, click **Quick download (1–2 mins)** or **Full download (5–10 mins)** in the left sidebar, or upload your own IMDb `.tsv.gz` files. Files are cached locally to avoid re-downloading.")

    app = UIApp()
    app.sidebar_data_loader()
    weights, min_votes, n_show = app.sidebar_settings()
    app.section_search_and_favs()
    app.section_recommendations(weights, min_votes, n_show)

    st.markdown("---")
    st.markdown(
        "Made by **Daniel Van Cuylenburg** · "
        "[LinkedIn](https://www.linkedin.com/in/daniel-van-cuylenburg-4770b518a/) · "
        "Data © IMDb"
    )

class UIApp:
    def __init__(self) -> None:
        self.downloader = DataDownloader()
        self.dataset = IMDBDataset()
        self.state = AppState.new()
        if 'imdb_df' in st.session_state:
            self.dataset.df = st.session_state['imdb_df']
            self.dataset.genre_matrix = st.session_state.get('imdb_G')
            self.dataset.genre_labels = st.session_state.get('imdb_genres', [])
            self.dataset.global_mean = st.session_state.get('imdb_C', 6.5)
            self.dataset.cast_map = st.session_state.get('imdb_cast_map', {})
            self.dataset.crew_directors_map = st.session_state.get('imdb_dir_map', {})
            self.dataset.crew_writers_map = st.session_state.get('imdb_wri_map', {})

    def sidebar_data_loader(self) -> None:
        st.sidebar.header("Data")
        mode = st.sidebar.radio("Load IMDb data via…", ("Download automatically", "Upload files"), index=0)

        basics_df = ratings_df = principals_df = crew_df = None

        if mode == "Upload files":
            up_basics = st.sidebar.file_uploader("Upload title.basics.tsv.gz", type=["gz"])
            up_ratings = st.sidebar.file_uploader("Upload title.ratings.tsv.gz", type=["gz"])
            up_principals = st.sidebar.file_uploader("(Optional) Upload title.principals.tsv.gz", type=["gz"])
            up_crew = st.sidebar.file_uploader("(Optional) Upload title.crew.tsv.gz", type=["gz"])
            if up_basics is not None and up_ratings is not None:
                with st.spinner("Reading uploaded files…"):
                    basics_df = IMDBDataset._read_tsv_gz_any(up_basics)
                    ratings_df = IMDBDataset._read_tsv_gz_any(up_ratings)
                    principals_df = IMDBDataset._read_tsv_gz_any(up_principals) if up_principals else None
                    crew_df = IMDBDataset._read_tsv_gz_any(up_crew) if up_crew else None
        else:
            st.sidebar.caption("**Quick** fetches basics+ratings; **Full** also pulls cast & crew.")
            c1, c2 = st.sidebar.columns(2)
            quick = c1.button("Quick download (1–2 mins)")
            full = c2.button("Full download (5–10 mins)")
            if quick or full:
                try:
                    with st.spinner("Downloading datasets from IMDb…"):
                        self.dataset.load_from_web(self.downloader, include_cast_crew=full)
                        st.session_state['imdb_df'] = self.dataset.df
                        st.session_state['imdb_G'] = self.dataset.genre_matrix
                        st.session_state['imdb_genres'] = self.dataset.genre_labels
                        st.session_state['imdb_C'] = self.dataset.global_mean
                        st.session_state['imdb_cast_map'] = self.dataset.cast_map
                        st.session_state['imdb_dir_map'] = self.dataset.crew_directors_map
                        st.session_state['imdb_wri_map'] = self.dataset.crew_writers_map
                        st.success("Datasets ready.")
                except Exception as e:
                    st.error(f"Download failed: {e}")

        if basics_df is not None and ratings_df is not None:
            with st.spinner("Preprocessing…"):
                self.dataset._finalise(basics_df, ratings_df, principals_df, crew_df)
                st.session_state['imdb_df'] = self.dataset.df
                st.session_state['imdb_G'] = self.dataset.genre_matrix
                st.session_state['imdb_genres'] = self.dataset.genre_labels
                st.session_state['imdb_C'] = self.dataset.global_mean
                st.session_state['imdb_cast_map'] = self.dataset.cast_map
                st.session_state['imdb_dir_map'] = self.dataset.crew_directors_map
                st.session_state['imdb_wri_map'] = self.dataset.crew_writers_map

    def sidebar_settings(self) -> Tuple[Weights, int, int]:
        st.sidebar.header("Recommendation settings")
        min_votes = int(st.sidebar.number_input("Minimum votes", min_value=0, value=DEFAULT_MIN_VOTES, step=100))
        n_show = int(st.sidebar.slider("How many recommendations?", 5, MAX_SHOW, DEFAULT_SHOW))
        with st.sidebar.expander("Weights", expanded=True):
            w_genre = float(st.slider("Weight: Genre", 0.0, 1.0, 0.5, 0.05))
            w_rating = float(st.slider("Weight: Rating", 0.0, 1.0, 0.2, 0.05))
            w_cast = float(st.slider("Weight: Cast overlap", 0.0, 1.0, 0.2, 0.05))
            w_crew = float(st.slider("Weight: Director/Writer", 0.0, 1.0, 0.1, 0.05))
        weights = Weights(w_genre, w_rating, w_cast, w_crew).normalised()

        st.sidebar.header("Quick filters")
        df = self.dataset.df
        if df is not None:
            min_year = int(pd.Series(df["startYear"]).dropna().min()) if not df["startYear"].dropna().empty else 1900
            max_year = int(pd.Series(df["startYear"]).dropna().max()) if not df["startYear"].dropna().empty else 2025
            year_range = st.sidebar.slider("Year range", min_year, max_year, (max(min_year, 1970), max_year))
            runtime_max = int(pd.Series(df["runtimeMinutes"]).dropna().max()) if "runtimeMinutes" in df.columns else 300
            runtime_range = st.sidebar.slider("Runtime (mins)", 0, max(60, min(runtime_max, 300)), (0, min(runtime_max, 300)))
            st.session_state["filter_year_range"] = year_range
            st.session_state["filter_runtime_range"] = runtime_range
        else:
            st.session_state["filter_year_range"] = None
            st.session_state["filter_runtime_range"] = None

        st.sidebar.header("Metadata & Posters (OMDb)")
        api_key = st.sidebar.text_input("OMDb API Key (optional)", type="password")
        st.session_state["omdb_api_key"] = api_key.strip() if api_key else ""
        st.session_state["use_metadata"] = bool(st.session_state["omdb_api_key"]) and st.sidebar.checkbox("Enable posters/plot & metadata filters", value=False)

        if st.session_state.get("use_metadata"):
            st.sidebar.caption("Filters apply to items with fetched metadata; click 'Refresh metadata' in Recommendations.")
            st.session_state["f_langs"] = st.sidebar.text_input("Languages (comma-separated, optional)")
            st.session_state["f_countries"] = st.sidebar.text_input("Countries (comma-separated, optional)")
            st.session_state["f_cert"] = st.sidebar.text_input("Certificate/Rated (e.g., PG-13, R)")

        return weights, min_votes, n_show

    def section_search_and_favs(self) -> None:
        st.subheader("🔎 Search titles")
        self.state.search = st.text_input("Search by title (case-insensitive)", self.state.search)
        rows_to_show = 200

        df = self.dataset.df
        if df is None:
            st.info("⬅️ Load the IMDb datasets first (upload or download). Then use the search below.")
            return

        query = self.state.search.strip()
        mask = (
            (
                df["primaryTitle"].str.contains(query, case=False, regex=False, na=False)
                | df["originalTitle"].str.contains(query, case=False, regex=False, na=False)
            )
            if query
            else pd.Series(True, index=df.index)
        )
        results = df.loc[mask, ["tconst", "display_title", "startYear", "genres", "rating", "votes"]].head(rows_to_show)

        if query and results.empty:
            st.caption("No matches. Tip: try the original title or broaden your search.")

        options = results.apply(
            lambda r: f"{r['display_title']} • ⭐ {r['rating']:.1f} ({r['votes']:,}) • {r['genres'].replace(',', ', ')}",
            axis=1,
        ).tolist()
        opt_to_tconst = {opt: t for opt, t in zip(options, results["tconst"])}

        selected = st.multiselect("Add to favourites (multi-select)", options, default=[])
        for opt in selected:
            self.state.favs.add(opt_to_tconst[opt])

        st.subheader("❤️ Your favourites")
        if self.state.favs:
            fav_df = (
                df.loc[df["tconst"].isin(self.state.favs), ["tconst", "display_title", "genres", "rating", "votes"]]
                .sort_values("display_title")
                .copy()
            )
            fav_df["rating"] = pd.to_numeric(fav_df["rating"], errors="coerce")
            fav_df["votes"] = pd.to_numeric(fav_df["votes"], errors="coerce").fillna(0).astype(int)

            api_key = st.session_state.get("omdb_api_key", "")
            show_ratings_votes = bool(api_key)

            nice_fav = fav_df.assign(genres=fav_df["genres"].str.replace(",", ", ")).rename(columns={
                "display_title": "Title",
                "genres": "Genres",
                "rating": "Rating",
                "votes": "Votes",
            })
            cols = ["Title", "Genres"] if not show_ratings_votes else ["Title", "Genres", "Rating", "Votes"]
            st.dataframe(nice_fav[cols], use_container_width=True)

            c1, c2, _ = st.columns(3)
            with c1:
                if st.button("Clear favourites"):
                    self.state.favs = set()
            with c2:
                st.download_button(
                    "Download favourites.csv",
                    nice_fav[cols].to_csv(index=False).encode("utf-8"),
                    file_name="favourites.csv",
                    mime="text/csv",
                )
            if not show_ratings_votes:
                st.caption("Add an OMDb API key in the sidebar if you'd like to display ratings and votes for your favourites.")
        else:
            st.info("No favourites yet — search above and add some.")

    def section_recommendations(self, weights: Weights, min_votes: int, n_show: int) -> None:
        st.subheader("🧠 Recommendations")
        if not self.state.favs:
            st.warning("Add at least one favourite to get recommendations.")
            return

        df = self.dataset.df
        yr = st.session_state.get("filter_year_range")
        rt = st.session_state.get("filter_runtime_range")
        cand_mask = np.ones(len(df), dtype=bool)
        if yr:
            y = df["startYear"].fillna(-1).astype(int)
            cand_mask &= (y >= yr[0]) & (y <= yr[1])
        if rt and "runtimeMinutes" in df.columns:
            r = df["runtimeMinutes"].fillna(-1).astype(int)
            cand_mask &= (r >= rt[0]) & (r <= rt[1])

        rec = Recommender(self.dataset, weights, min_votes=min_votes)
        out = rec.recommend(self.state.favs, k=n_show, candidate_mask=cand_mask)
        if out.empty:
            st.info("No recommendations match your filters (try lowering the minimum votes or widening filters).")
            return

        meta_enabled = st.session_state.get("use_metadata")
        api_key = st.session_state.get("omdb_api_key", "")

        def _fetch_meta(tconst: str) -> Optional[dict]:
            @st.cache_data(show_spinner=False)
            def _call(imdb_id: str, key: str) -> Optional[dict]:
                if not key:
                    return None
                import requests
                try:
                    resp = requests.get("https://www.omdbapi.com/", params={"apikey": key, "i": imdb_id}, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data if data.get("Response") == "True" else None
                except Exception:
                    return None
                return None
            return _call(tconst, api_key)

        def _norm_list(csv_text: str) -> list[str]:
            return [x.strip().lower() for x in csv_text.split(',') if x.strip()] if csv_text else []

        if meta_enabled and api_key:
            need_meta = out["tconst"].tolist()
            metas: dict[str, dict] = {}
            for t in need_meta:
                m = _fetch_meta(t)
                if m:
                    metas[t] = m
            langs = _norm_list(st.session_state.get("f_langs", ""))
            countries = _norm_list(st.session_state.get("f_countries", ""))
            cert = st.session_state.get("f_cert", "").strip().lower()
            if langs or countries or cert:
                def _keep(t: str) -> bool:
                    m = metas.get(t)
                    if not m:
                        return False
                    ok = True
                    if langs:
                        ok &= any(l.strip().lower() in m.get("Language", "").lower() for l in langs)
                    if countries:
                        ok &= any(c.strip().lower() in m.get("Country", "").lower() for c in countries)
                    if cert:
                        ok &= cert in m.get("Rated", "").lower()
                    return ok
                mask = out["tconst"].apply(_keep)
                out = out[mask]
                if out.empty:
                    st.info("No items match the metadata filters. Clear them to see recommendations.")
                    return

        nice_out = out.copy()
        nice_out["genres"] = nice_out["genres"].str.replace(",", ", ")
        nice_out = nice_out.rename(columns={
            "display_title": "Title",
            "genres": "Genres",
            "startYear": "Year",
            "rating": "Rating",
            "votes": "Votes",
            "score": "Score",
        })[["Title", "Genres", "Year", "Rating", "Votes", "Score"]]
        nice_out["Score"] = nice_out["Score"].round(3)
        st.dataframe(nice_out, use_container_width=True)
        st.download_button(
            "Download recommendations.csv",
            nice_out.to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv",
            mime="text/csv",
        )

        with st.expander("📍 2D map of current recommendations"):
            if st.button("Compute 2D map"):
                try:
                    df_all = self.dataset.df
                    t_to_idx = {t: i for i, t in enumerate(df_all["tconst"].tolist())}
                    idxs = [t_to_idx[t] for t in out["tconst"].tolist() if t in t_to_idx]
                    G = self.dataset.genre_matrix[idxs]
                    yr = df_all.iloc[idxs]["startYear"].fillna(0).to_numpy(np.float32).reshape(-1, 1)
                    rt = df_all.iloc[idxs]["rating"].fillna(0).to_numpy(np.float32).reshape(-1, 1)
                    def _mm(a: np.ndarray) -> np.ndarray:
                        a = a.astype(np.float32)
                        mn, mx = float(a.min()), float(a.max())
                        return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a, dtype=np.float32)
                    X = np.hstack([G, _mm(yr), _mm(rt)])
                    emb = TSNE(n_components=2, perplexity=max(5, min(30, len(idxs)//3 or 5)), learning_rate='auto', init='random', random_state=42).fit_transform(X)
                    pts = pd.DataFrame({
                        "x": emb[:, 0],
                        "y": emb[:, 1],
                        "Title": out["display_title"].tolist(),
                        "Fav": out["tconst"].isin(list(self.state.favs)).map({True: "Favourite", False: "Recommended"}).tolist(),
                        "Rating": out["rating"].tolist(),
                        "Year": out["startYear"].tolist(),
                    })
                    import altair as alt
                    chart = alt.Chart(pts).mark_circle(size=80).encode(
                        x=alt.X('x:Q', title='t-SNE 1'),
                        y=alt.Y('y:Q', title='t-SNE 2'),
                        color='Fav:N',
                        tooltip=['Title:N','Year:Q','Rating:Q']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not compute map: {e}")
