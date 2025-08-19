# IMDb Favourites-Based Movie Recommender (Streamlit, OOP)

A lightweight, production-style Streamlit app that lets you:
- **Search** IMDb movies and mark **favourites**.
- Get **recommendations** based on genre similarity, IMDb rating prior, **cast overlap**, and **director/writer** similarity (when available).
- Optionally **pull posters/plots** and filter by **language/country/certificate** via the **OMDb** API.

---

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate  # (Windows PowerShell)
pip install -U streamlit pandas numpy scikit-learn requests altair
streamlit run app.py
```

> **Tip:** If you want a speedier first run, click **Quick download (1–2 mins)**; for full hybrid recommendations (with cast/crew), choose **Full download (5–10 mins)**. Files are cached in `~/.cache/imdb_datasets` to avoid re-downloading.

---

## Project layout

```
app.py                          # Streamlit entrypoint
imdb_recommender/
  __init__.py
  config.py                     # URLs, defaults, cache/chunk settings
  utils.py                      # small helpers
  downloader.py                 # cached, parallel downloader
  dataset.py                    # IMDb loaders, preprocessing, feature builds
  recommender.py                # Weights + hybrid recommender
  ui.py                         # Streamlit UI (sidebar, search, favourites, recs)
```

---

## Features

- **Cold start**: “Quick download” uses just `title.basics` + `title.ratings` for fast setup. “Full download” also fetches `title.principals` + `title.crew` to enable cast and creator overlaps.
- **Hybrid scoring** (weights configurable):
  - Genre cosine similarity.
  - Bayesian-weighted rating prior (adjusted for vote counts).
  - Cast overlap (actors/actresses).
  - Director/Writer overlap.
- **Quick filters**: year range, runtime; optional metadata filters (language/country/certificate) when OMDb is enabled.
- **2D map**: t-SNE map of recommendations (genre + year + rating), favourites highlighted.
- **Posters/plots**: via OMDb if you provide an API key.

---

## OMDb (optional)

- Add your key in the left sidebar to enable posters/plots and metadata filtering.
- If no OMDb key is provided, the **favourites table hides Ratings/Votes** to avoid confusing zeroes.

Get a key at: https://www.omdbapi.com/

---

## Notes

- This project is not affiliated with IMDb or OMDb. Respect their dataset/API terms.
- The IMDb dataset files are large (hundreds of MB). On first run, downloads can take minutes depending on network speed. Subsequent runs use the local cache.
- Tested with Python 3.10+.
