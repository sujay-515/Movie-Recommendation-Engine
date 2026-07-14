import os
import re
import difflib
import pandas as pd
import numpy as np
import pickle
import time
import httpx
import logging
import asyncio
from dotenv import load_dotenv

# Load env variables from backend/.env
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path)

# Configure logger
logger = logging.getLogger("backend.recommender")

# ─── Global cache ─────────────────────────────────────────────────────────────
_movies_df = None          # Full movies DataFrame (49,503 rows)
_embeddings_matrix = None  # np.ndarray [N, 384], L2-normalized
_movie_index_df = None     # Aligned index: Title/Year/TMDb_Rating/vote_count/Genres/Keywords
_tmdb_cache = {}


# ─── TMDB helpers (unchanged from previous version) ───────────────────────────

def _parse_tmdb_response(details, keywords_data):
    """
    Common helper to parse rating, tags, and tag string from TMDB API responses.
    """
    kw_list = [kw["name"].lower().strip() for kw in keywords_data.get("keywords", [])]
    genres = [g["name"].lower().strip() for g in details.get("genres", [])]
    combined_tags = set(genres + kw_list)
    combined_tags_str = ", ".join(sorted(combined_tags))
    rating = details.get("vote_average", 0.0)
    return rating, combined_tags, combined_tags_str


def fetch_tmdb_data(title, year=None):
    """
    Search and fetch rating (vote_average) and keywords for a movie from TMDB.
    Uses an in-memory cache with 24-hour expiration.
    Returns: (rating, tag_set, tag_str) if success, else None
    """
    global _tmdb_cache
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.warning(f"[TMDB FALLBACK] '{title}' -> No TMDB API key configured.")
        return None

    cache_key = f"{title.lower().strip()}|{year or ''}"
    now = time.time()

    if cache_key in _tmdb_cache:
        cached = _tmdb_cache[cache_key]
        if now - cached["timestamp"] < 86400:
            logger.info(f"[TMDB CACHE HIT] '{title}' ({year})")
            return cached["rating"], cached["tag_set"], cached["tags_str"]

    logger.info(f"[TMDB CACHE MISS] '{title}' ({year}) -> Fetching from TMDB...")
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title.strip()}
    if year:
        params["primary_release_year"] = year

    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(search_url, params=params)
            resp.raise_for_status()
            search_data = resp.json()

            results = search_data.get("results", [])
            if not results:
                logger.warning(f"[TMDB FALLBACK] '{title}' ({year}) -> No results found.")
                _tmdb_cache[cache_key] = {
                    "rating": 0.0, "tag_set": set(), "tags_str": "", "timestamp": now
                }
                return None

            best_match = None
            for res in results:
                res_title = res.get("title", "")
                res_orig_title = res.get("original_title", "")
                sim_1 = difflib.SequenceMatcher(None, title.lower().strip(), res_title.lower().strip()).ratio()
                sim_2 = difflib.SequenceMatcher(None, title.lower().strip(), res_orig_title.lower().strip()).ratio()
                if max(sim_1, sim_2) >= 0.85:
                    best_match = res
                    break

            if not best_match:
                best_match = results[0]

            movie_id = best_match["id"]
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            details_resp = client.get(details_url, params={"api_key": api_key})
            details_resp.raise_for_status()
            keywords_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"
            keywords_resp = client.get(keywords_url, params={"api_key": api_key})
            keywords_resp.raise_for_status()

            rating, combined_tags, combined_tags_str = _parse_tmdb_response(
                details_resp.json(), keywords_resp.json()
            )

            _tmdb_cache[cache_key] = {
                "rating": rating,
                "tag_set": combined_tags,
                "tags_str": combined_tags_str,
                "timestamp": now
            }
            logger.info(f"[TMDB REFRESH SUCCESS] '{title}' ({year}) -> Rating: {rating}")
            return rating, combined_tags, combined_tags_str

    except Exception as e:
        logger.error(f"[TMDB FALLBACK] '{title}' ({year}) -> Query failed: {e}")
        return None


def refresh_movie_in_memory(title, movies_df):
    """
    Deprecated/Legacy compatibility stub.
    """
    target_rows = movies_df[movies_df["Title"].str.lower() == title.lower()]
    if target_rows.empty:
        return None
    idx = target_rows.index[0]
    row = movies_df.loc[idx]
    year = int(row["Year"]) if "Year" in movies_df.columns and not pd.isna(row["Year"]) else None
    res = fetch_tmdb_data(title, year)
    if res:
        rating, _, _ = res
        movies_df.at[idx, "TMDb_Rating"] = rating
        movies_df.at[idx, "IMDb_Rating"] = rating
    return movies_df.loc[idx]


async def fetch_tmdb_data_async(client: httpx.AsyncClient, title, year=None):
    """
    Search and fetch rating (vote_average) and keywords for a movie from TMDB asynchronously.
    Uses an in-memory cache with 24-hour expiration.
    """
    global _tmdb_cache
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        logger.warning(f"[TMDB FALLBACK] '{title}' -> No TMDB API key configured.")
        return None

    cache_key = f"{title.lower().strip()}|{year or ''}"
    now = time.time()

    if cache_key in _tmdb_cache:
        cached = _tmdb_cache[cache_key]
        if now - cached["timestamp"] < 86400:
            logger.info(f"[TMDB CACHE HIT] '{title}' ({year})")
            return cached["rating"], cached["tag_set"], cached["tags_str"]

    logger.info(f"[TMDB CACHE MISS] '{title}' ({year}) -> Fetching from TMDB asynchronously...")
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": title.strip()}
    if year:
        params["primary_release_year"] = year

    try:
        resp = await client.get(search_url, params=params, timeout=2.5)
        resp.raise_for_status()
        search_data = resp.json()

        results = search_data.get("results", [])
        if not results:
            logger.warning(f"[TMDB FALLBACK] '{title}' ({year}) -> No TMDB results found.")
            _tmdb_cache[cache_key] = {
                "rating": 0.0, "tag_set": set(), "tags_str": "", "timestamp": now
            }
            return None

        best_match = None
        for res in results:
            res_title = res.get("title", "")
            res_orig_title = res.get("original_title", "")
            sim_1 = difflib.SequenceMatcher(None, title.lower().strip(), res_title.lower().strip()).ratio()
            sim_2 = difflib.SequenceMatcher(None, title.lower().strip(), res_orig_title.lower().strip()).ratio()
            if max(sim_1, sim_2) >= 0.85:
                best_match = res
                break

        if not best_match:
            best_match = results[0]

        movie_id = best_match["id"]
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        keywords_url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords"

        details_resp, keywords_resp = await asyncio.gather(
            client.get(details_url, params={"api_key": api_key}, timeout=2.5),
            client.get(keywords_url, params={"api_key": api_key}, timeout=2.5)
        )

        details_resp.raise_for_status()
        keywords_resp.raise_for_status()

        rating, combined_tags, combined_tags_str = _parse_tmdb_response(
            details_resp.json(), keywords_resp.json()
        )

        _tmdb_cache[cache_key] = {
            "rating": rating,
            "tag_set": combined_tags,
            "tags_str": combined_tags_str,
            "timestamp": now
        }
        logger.info(f"[TMDB REFRESH SUCCESS] '{title}' ({year}) -> Rating: {rating}, Keywords: {len(combined_tags)}")
        return rating, combined_tags, combined_tags_str

    except Exception as e:
        logger.error(f"[TMDB FALLBACK] '{title}' ({year}) -> Query failed: {e}")
        return None


async def refresh_movie_in_memory_async(client: httpx.AsyncClient, title, movies_df):
    """
    Deprecated/Legacy compatibility stub.
    """
    target_rows = movies_df[movies_df["Title"].str.lower() == title.lower()]
    if target_rows.empty:
        return None
    idx = target_rows.index[0]
    row = movies_df.loc[idx]
    year = int(row["Year"]) if "Year" in movies_df.columns and not pd.isna(row["Year"]) else None
    res = await fetch_tmdb_data_async(client, title, year)
    if res:
        rating, _, _ = res
        movies_df.at[idx, "TMDb_Rating"] = rating
        movies_df.at[idx, "IMDb_Rating"] = rating
    return movies_df.loc[idx]


async def get_movie_enrichment_async(client: httpx.AsyncClient, title, year, movies_df):
    """
    Retrieves rating, genres, and keywords for a movie without mutating movies_df.
    Prefers live TMDB rating; always uses clean CSV Genres/Keywords for display accuracy.
    Returns: (rating: float, genres_str: str, keywords_str: str)
    """
    row = movies_df[movies_df["Title"].str.lower() == title.lower()]
    if row.empty:
        return 0.0, "", ""

    csv_rating = float(row.iloc[0]["TMDb_Rating"]) if pd.notna(row.iloc[0]["TMDb_Rating"]) else 0.0
    csv_genres = str(row.iloc[0]["Genres"]) if pd.notna(row.iloc[0]["Genres"]) else ""
    csv_keywords = (
        str(row.iloc[0]["Keywords"])
        if pd.notna(row.iloc[0]["Keywords"]) and str(row.iloc[0]["Keywords"]).strip()
        else ""
    )

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return csv_rating, csv_genres, csv_keywords

    res = await fetch_tmdb_data_async(client, title, year)
    if res:
        rating, _, _ = res  # Use TMDB for rating; keep CSV Genres/Keywords for display accuracy
        return rating, csv_genres, csv_keywords

    return csv_rating, csv_genres, csv_keywords


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_movies(filename=None):
    """
    Loads and caches the movies DataFrame and pre-built sentence embedding index.

    Expects:
      - data/top50k_movies.csv  (49,503 rows; columns: Title, Year, TMDb_Rating,
                                  vote_count, popularity, Overview, Genres, Keywords)
      - backend/movie_embeddings.npy  (shape [N, 384], L2-normalized float32)
      - backend/movie_index.pkl       (aligned DataFrame; run build_embeddings.py to create)

    The IMDb_Rating alias is added for backward compat with the main.py search endpoint
    which reads that column name. It is read-only and never written.
    """
    global _movies_df, _embeddings_matrix, _movie_index_df

    if _movies_df is not None:
        return _movies_df

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "..", "data", "top50k_movies.csv"),
            "data/top50k_movies.csv",
            "../data/top50k_movies.csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filename = path
                break
        else:
            raise FileNotFoundError("Could not find 'data/top50k_movies.csv' in any expected location.")

    try:
        movies = pd.read_csv(filename, on_bad_lines="skip")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")

    if movies.empty:
        raise ValueError("No movies found in the file.")

    required_cols = ["Title", "TMDb_Rating", "Year", "Genres", "Overview"]
    for col in required_cols:
        if col not in movies.columns:
            if col == "Year":
                movies["Year"] = 0
            elif col == "TMDb_Rating":
                movies["TMDb_Rating"] = 0.0
            else:
                raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Read-only alias so the main.py search endpoint (which uses "IMDb_Rating") doesn't break
    movies["IMDb_Rating"] = movies["TMDb_Rating"]

    # Ensure Keywords column exists; fill NaN with empty string
    if "Keywords" not in movies.columns:
        movies["Keywords"] = ""
    else:
        movies["Keywords"] = movies["Keywords"].fillna("")

    # Load pre-built embedding cache
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(script_dir, "movie_embeddings.npy")
    index_path = os.path.join(script_dir, "movie_index.pkl")

    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        _embeddings_matrix = np.load(embeddings_path)
        with open(index_path, "rb") as f:
            _movie_index_df = pickle.load(f)
        logger.info(
            f"[Embeddings] Loaded: shape={_embeddings_matrix.shape}, "
            f"index={len(_movie_index_df)} rows."
        )
    else:
        logger.warning(
            "[Embeddings] Cache not found. Run: python backend/build_embeddings.py"
        )
        _embeddings_matrix = None
        _movie_index_df = None

    _movies_df = movies
    return _movies_df


def find_closest_title(user_input, movies_df):
    """
    Finds the single closest movie title match from the dataframe using difflib.
    """
    all_titles = movies_df["Title"].values
    matches = difflib.get_close_matches(user_input, all_titles, n=1, cutoff=0.7)
    return matches[0] if matches else None


def build_blob(row: pd.Series) -> str:
    """
    Build the text blob used for sentence embedding for a single movie row.

    Format:
      "({Year}). Genres: {Genres}. {Overview}[. Keywords: {Keywords}.]"

    Rules:
      - Title is EXCLUDED from the embedded text block to prevent spurious matches
        driven by word overlap (e.g., 'Donnie Brasco' for 'Donnie Darko').
      - Overview is the dominant semantic content.
      - If Keywords is empty/NaN the 'Keywords:' clause is omitted entirely.
    """
    try:
        year = str(int(row["Year"])) if not pd.isna(row["Year"]) else ""
    except (ValueError, TypeError):
        year = ""
    genres = str(row["Genres"]).strip() if pd.notna(row["Genres"]) else ""
    overview = str(row["Overview"]).strip() if pd.notna(row["Overview"]) else ""
    keywords_raw = str(row["Keywords"]).strip() if pd.notna(row["Keywords"]) else ""

    blob = f"({year}). Genres: {genres}. {overview}"
    if keywords_raw:
        blob += f". Keywords: {keywords_raw}."
    else:
        blob += "."
    return blob


# ─── Embedding-based scoring ──────────────────────────────────────────────────

def scale_to_display(raw):
    """
    Maps raw cosine similarity scores to a display-friendly percentage band.
    Calibrated to the all-MiniLM-L6-v2 distribution on this 49,503-movie corpus:

      Observed range across test anchors (Avatar, Interstellar, Inception, etc.):
        Top match:  0.57 – 0.71
        p99:        0.42 – 0.47
        p95:        0.28 – 0.37
        p90:        0.25 – 0.33

      Display bands:
        ≥ 0.55  →  92–98%  (very similar — same franchise/plot/theme)
        ≥ 0.42  →  83–91%  (strongly related)
        ≥ 0.30  →  75–82%  (thematically related)
    """
    if raw >= 0.55:
        return round(0.92 + min((raw - 0.55) / 0.20, 1.0) * 0.06, 4)  # 92–98%
    elif raw >= 0.42:
        return round(0.83 + ((raw - 0.42) / 0.13) * 0.09, 4)          # 83–91%
    else:
        return round(0.75 + ((raw - 0.30) / 0.12) * 0.08, 4)          # 75–82%


async def get_recommendations_async(closest_title, movies_df, top_n=5):
    """
    Finds the top N recommended movies using pre-computed sentence embedding similarity.

    Algorithm:
      1. Look up the anchor movie's L2-normalized embedding vector.
      2. Compute cosine similarity via a single vectorized dot product:
         raw_scores = embeddings_matrix @ anchor_vec  (shape [N])
      3. Apply fractional genre factor:
           factor = 0.82 + 0.38 * (shared_genres / anchor_genres)
         This scales from a penalty of 0.82x up to a bonus of 1.20x depending on
         the proportion of anchor genres that the candidate shares.
      4. Apply vote_count soft multiplier (log10-based, [0.85, 1.10]).
      5. Clamp final score to 1.0. Exclude anchor itself.
      6. Filter below SIMILARITY_THRESHOLD. Sort: [score DESC, vote_count DESC, TMDb_Rating DESC].
      7. Async-enrich top_n with live TMDB ratings.
    """
    global _embeddings_matrix, _movie_index_df

    if _embeddings_matrix is None or _movie_index_df is None:
        logger.warning("[Recommendations] Embeddings not loaded — returning empty. Run build_embeddings.py.")
        return []

    title_lower = closest_title.lower().strip()
    anchor_mask = _movie_index_df["Title"].str.lower().str.strip() == title_lower
    if not anchor_mask.any():
        logger.warning(f"[Recommendations] '{closest_title}' not found in embedding index.")
        return []

    anchor_idx = int(_movie_index_df.index[anchor_mask][0])
    anchor_vec = _embeddings_matrix[anchor_idx]  # [384], already L2-normalized

    # Step 1: vectorized cosine similarity (dot product since vectors are pre-normalized)
    raw_scores = _embeddings_matrix @ anchor_vec  # [N]

    # Step 2a: fractional genre factor — scales from 0.82x penalty up to 1.20x bonus.
    anchor_genres_str = str(_movie_index_df.loc[anchor_idx, "Genres"])
    anchor_genres = {g.strip().lower() for g in anchor_genres_str.split(",") if g.strip()}

    if anchor_genres:
        # Calculate overlap for every row efficiently
        genres_series = _movie_index_df["Genres"].fillna("")
        
        # Since calculating this over 49k rows on-the-fly using series.apply is fast:
        def compute_factor(val):
            cg = {g.strip().lower() for g in str(val).split(",") if g.strip()}
            overlap = len(anchor_genres & cg)
            return 0.82 + 0.38 * (overlap / len(anchor_genres))

        genre_factor = genres_series.apply(compute_factor).values
    else:
        genre_factor = np.ones(len(raw_scores))

    # Step 2b: vote_count soft multiplier — log10-based, bounded [0.85, 1.10].
    _vc = np.clip(_movie_index_df["vote_count"].values.astype(float), 1, None)
    _vc_log = np.log10(_vc)
    _vc_min_log = np.log10(20)      # dataset floor
    _vc_max_log = np.log10(50_000)  # ~99th percentile
    _vc_norm = np.clip((_vc_log - _vc_min_log) / (_vc_max_log - _vc_min_log), 0.0, 1.0)
    vc_factor = 0.85 + _vc_norm * 0.25

    final_scores = np.minimum(raw_scores * genre_factor * vc_factor, 1.0)

    # Step 3: build candidate DataFrame, exclude anchor
    cands = _movie_index_df.copy()
    cands["_raw"] = raw_scores
    cands["_score"] = final_scores
    cands = cands[cands.index != anchor_idx]

    # Step 4: threshold filter
    SIMILARITY_THRESHOLD = 0.30
    cands = cands[cands["_score"] >= SIMILARITY_THRESHOLD]


    # Step 5: sort and take top_n
    cands = cands.sort_values(
        by=["_score", "vote_count", "TMDb_Rating"],
        ascending=[False, False, False]
    ).head(top_n).copy()

    cands["RawSimilarity"] = cands["_score"]
    cands["Similarity"] = cands["_score"].apply(scale_to_display)

    # Step 6: async TMDB enrichment for top_n candidates
    refreshed = []
    if not cands.empty:
        async with httpx.AsyncClient() as client:
            tasks = [
                get_movie_enrichment_async(
                    client,
                    str(row["Title"]),
                    int(row["Year"]) if not pd.isna(row["Year"]) else None,
                    movies_df
                )
                for _, row in cands.iterrows()
            ]
            enrichments = await asyncio.gather(*tasks)

        for (_, row), (enriched_rating, _, _) in zip(cands.iterrows(), enrichments):
            refreshed.append({
                "Title": str(row["Title"]),
                "IMDb_Rating": enriched_rating,
                "Year": int(row["Year"]) if not pd.isna(row["Year"]) else 0,
                "Similarity": float(row["Similarity"]),
                "RawSimilarity": float(row["RawSimilarity"])
            })

    return refreshed


def get_recommendations(closest_title, movies_df, top_n=5):
    """
    Sync fallback version. Uses embedding similarity with sync TMDB rating enrichment.
    """
    global _embeddings_matrix, _movie_index_df

    if _embeddings_matrix is None or _movie_index_df is None:
        logger.warning("[Recommendations] Embeddings not loaded.")
        return []

    title_lower = closest_title.lower().strip()
    anchor_mask = _movie_index_df["Title"].str.lower().str.strip() == title_lower
    if not anchor_mask.any():
        return []

    anchor_idx = int(_movie_index_df.index[anchor_mask][0])
    anchor_vec = _embeddings_matrix[anchor_idx]
    raw_scores = _embeddings_matrix @ anchor_vec

    anchor_genres_str = str(_movie_index_df.loc[anchor_idx, "Genres"])
    anchor_genres = {g.strip().lower() for g in anchor_genres_str.split(",") if g.strip()}

    if anchor_genres:
        genres_series = _movie_index_df["Genres"].fillna("")
        has_overlap = genres_series.apply(
            lambda gs: bool(anchor_genres & {g.strip().lower() for g in str(gs).split(",") if g.strip()})
        ).values
        genre_factor = np.where(has_overlap, 1.20, 0.85)
    else:
        genre_factor = np.ones(len(raw_scores))

    _vc = np.clip(_movie_index_df["vote_count"].values.astype(float), 1, None)
    _vc_log = np.log10(_vc)
    _vc_min_log = np.log10(20)
    _vc_max_log = np.log10(50_000)
    _vc_norm = np.clip((_vc_log - _vc_min_log) / (_vc_max_log - _vc_min_log), 0.0, 1.0)
    vc_factor = 0.85 + _vc_norm * 0.25

    final_scores = np.minimum(raw_scores * genre_factor * vc_factor, 1.0)


    cands = _movie_index_df.copy()
    cands["_score"] = final_scores
    cands = cands[cands.index != anchor_idx]
    cands = cands[cands["_score"] >= 0.30]
    cands = cands.sort_values(
        by=["_score", "vote_count", "TMDb_Rating"],
        ascending=[False, False, False]
    ).head(top_n).copy()

    cands["RawSimilarity"] = cands["_score"]
    cands["Similarity"] = cands["_score"].apply(scale_to_display)

    results = []
    for _, row in cands.iterrows():
        year = int(row["Year"]) if not pd.isna(row["Year"]) else None
        enriched_rating = float(row["TMDb_Rating"]) if not pd.isna(row["TMDb_Rating"]) else 0.0
        api_key = os.getenv("TMDB_API_KEY")
        if api_key:
            res = fetch_tmdb_data(str(row["Title"]), year)
            if res:
                enriched_rating, _, _ = res

        results.append({
            "Title": str(row["Title"]),
            "IMDb_Rating": enriched_rating,
            "Year": int(row["Year"]) if not pd.isna(row["Year"]) else 0,
            "Similarity": float(row["Similarity"]),
            "RawSimilarity": float(row["RawSimilarity"])
        })

    return results
