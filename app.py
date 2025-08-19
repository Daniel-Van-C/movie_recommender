"""
Streamlit entrypoint for the IMDb Favourites-Based Movie Recommender.
Run:
    pip install -U streamlit pandas numpy scikit-learn requests altair
    streamlit run app.py
"""
from imdb_recommender.ui import main

if __name__ == "__main__":
    main()
