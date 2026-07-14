import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from backend.main import app
import backend.recommender as recommender
from backend.recommender import (
    find_closest_title,
    get_recommendations_async,
    get_movie_enrichment_async,
    scale_to_display,
    build_blob,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_movies_df():
    """Minimal DataFrame matching the new top50k_movies.csv column schema."""
    df = pd.DataFrame([
        {
            "Title": "Inception",
            "Year": 2010,
            "TMDb_Rating": 8.4,
            "vote_count": 34495,
            "popularity": 83.9,
            "Overview": "A skilled thief commits corporate espionage by infiltrating dreams.",
            "Genres": "Action, Science Fiction, Adventure",
            "Keywords": "dream, virtual reality, subconscious, manipulation",
        },
        {
            "Title": "Interstellar",
            "Year": 2014,
            "TMDb_Rating": 8.4,
            "vote_count": 32000,
            "popularity": 140.2,
            "Overview": "Explorers use a wormhole to travel beyond the galaxy.",
            "Genres": "Adventure, Drama, Science Fiction",
            "Keywords": "space travel, wormhole, time warp, nasa",
        },
        {
            "Title": "The Dark Knight",
            "Year": 2008,
            "TMDb_Rating": 8.5,
            "vote_count": 30000,
            "popularity": 130.6,
            "Overview": "Batman fights the Joker in Gotham City.",
            "Genres": "Action, Drama, Crime, Thriller",
            "Keywords": "joker, vigilante, batman, chaos",
        },
        {
            "Title": "Toy Story",
            "Year": 1995,
            "TMDb_Rating": 7.9,
            "vote_count": 15000,
            "popularity": 60.0,
            "Overview": "A cowboy toy is threatened by a new space ranger toy.",
            "Genres": "Animation, Adventure, Comedy, Family",
            "Keywords": "toy, friendship, jealousy, adventure",
        },
    ])
    df["IMDb_Rating"] = df["TMDb_Rating"]
    df["Keywords"] = df["Keywords"].fillna("")
    return df


@pytest.fixture
def mock_embeddings(mock_movies_df):
    """
    Returns a small fake L2-normalized embeddings matrix (4 × 384) and the aligned index_df.
    Inception and Interstellar share a genre (Science Fiction), so their embeddings
    are set to be more similar.
    """
    n = len(mock_movies_df)
    dim = 384
    rng = np.random.default_rng(42)
    E = rng.standard_normal((n, dim)).astype(np.float32)
    # Make Inception (0) and Interstellar (1) intentionally similar
    E[1] = E[0] * 0.9 + rng.standard_normal(dim).astype(np.float32) * 0.1
    # L2-normalize each row
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / norms
    index_df = mock_movies_df[["Title", "Year", "TMDb_Rating", "vote_count", "Genres", "Keywords"]].copy()
    index_df = index_df.reset_index(drop=True)
    return E, index_df


@pytest.fixture(autouse=True)
def reset_recommender_globals(mock_embeddings, mock_movies_df):
    """Inject mock embeddings and index; clear TMDB cache before each test."""
    E, index_df = mock_embeddings
    recommender._movies_df = mock_movies_df
    recommender._embeddings_matrix = E
    recommender._movie_index_df = index_df
    recommender._tmdb_cache.clear()
    yield
    recommender._movies_df = None
    recommender._embeddings_matrix = None
    recommender._movie_index_df = None


# ─── Blob format tests ────────────────────────────────────────────────────────

class TestBuildBlob:
    def test_blob_with_keywords(self):
        row = pd.Series({
            "Title": "Inception",
            "Year": 2010,
            "Genres": "Action, Science Fiction",
            "Overview": "A thief who enters dreams.",
            "Keywords": "dream, heist",
        })

        blob = build_blob(row)
        assert blob.startswith("(2010).")
        assert "Inception" not in blob  # Title must be excluded to prevent word overlap bias
        assert "Genres: Action, Science Fiction" in blob
        assert "A thief who enters dreams." in blob
        assert "Keywords: dream, heist." in blob

    def test_blob_empty_keywords_omits_clause(self):
        """When Keywords is NaN or empty, the 'Keywords:' clause must not appear at all."""
        row_nan = pd.Series({
            "Title": "The Kissing Booth 2",
            "Year": 2020,
            "Genres": "Comedy, Romance",
            "Overview": "Elle juggles college decisions and romance.",
            "Keywords": float("nan"),
        })
        blob = build_blob(row_nan)
        assert "Keywords" not in blob
        assert blob.endswith(".")

    def test_blob_empty_string_keywords_omits_clause(self):
        row_empty = pd.Series({
            "Title": "Some Film",
            "Year": 2015,
            "Genres": "Drama",
            "Overview": "A dramatic story.",
            "Keywords": "",
        })
        blob = build_blob(row_empty)
        assert "Keywords" not in blob

    def test_blob_title_not_present(self):
        row = pd.Series({
            "Title": "Avatar",
            "Year": 2009,
            "Genres": "Action, Adventure",
            "Overview": "A paraplegic Marine on Pandora.",
            "Keywords": "alien, future",
        })
        blob = build_blob(row)
        assert "Avatar" not in blob


# ─── Genre bonus tests ────────────────────────────────────────────────────────

class TestGenreBonus:
    def test_fractional_genre_bonus_for_partial_match(self):
        """
        If anchor has 3 genres (Action, SciFi, Adventure) and candidate has 2 matches (SciFi, Adventure),
        the fractional factor must be 0.82 + 0.38 * (2/3) = 1.0733.
        """
        anchor_genres = {"action", "science fiction", "adventure"}
        candidate_genres = {"adventure", "drama", "science fiction"}
        overlap = len(anchor_genres & candidate_genres)  # 2
        factor = 0.82 + 0.38 * (overlap / len(anchor_genres))
        assert abs(factor - 1.0733) < 0.001

    def test_genre_mismatch_penalty(self):
        """
        When there is 0 genre overlap, the factor must be the floor (0.82x).
        """
        anchor_genres = {"action", "science fiction", "adventure"}
        candidate_genres = {"animation", "comedy", "family"}
        overlap = len(anchor_genres & candidate_genres)  # 0
        factor = 0.82 + 0.38 * (overlap / len(anchor_genres))
        assert factor == 0.82

    def test_score_clamped_at_1(self):
        """
        Even with a maximum 1.20x genre factor, the final score is capped at 1.0.
        """
        raw = 0.95
        factor = 1.20
        clamped = min(raw * factor, 1.0)
        assert clamped == 1.0

    def test_score_not_clamped_when_below_1(self):
        raw = 0.60
        factor = 1.0733
        expected = raw * factor  # 0.64398
        result = min(raw * factor, 1.0)
        assert abs(result - expected) < 1e-6


# ─── Scale to display tests ───────────────────────────────────────────────────

class TestScaleToDisplay:
    def test_high_similarity_maps_to_92_to_98(self):
        # raw >= 0.55 → 92–98%
        assert 0.92 <= scale_to_display(0.60) <= 0.98
        assert 0.92 <= scale_to_display(0.70) <= 0.98

    def test_mid_similarity_maps_to_83_to_91(self):
        # 0.42 <= raw < 0.55 → 83–91%
        assert 0.83 <= scale_to_display(0.45) <= 0.91
        assert 0.83 <= scale_to_display(0.52) <= 0.91

    def test_low_similarity_maps_to_75_to_82(self):
        # 0.30 <= raw < 0.42 → 75–82%
        assert 0.75 <= scale_to_display(0.32) <= 0.82
        assert 0.75 <= scale_to_display(0.40) <= 0.82

    def test_scale_is_monotone(self):
        """Higher raw similarity should always map to higher display score."""
        scores = [0.31, 0.36, 0.43, 0.50, 0.58, 0.65]
        display = [scale_to_display(s) for s in scores]
        assert display == sorted(display)



# ─── Title matching tests ──────────────────────────────────────────────────────

class TestTitleMatching:
    def test_exact_match(self, mock_movies_df):
        assert find_closest_title("Inception", mock_movies_df) == "Inception"

    def test_fuzzy_match(self, mock_movies_df):
        assert find_closest_title("Incepton", mock_movies_df) == "Inception"

    def test_no_match_returns_none(self, mock_movies_df):
        assert find_closest_title("Completely Unknown Title XYZ", mock_movies_df) is None


# ─── API endpoint smoke tests ─────────────────────────────────────────────────

class TestAPIEndpoints:
    def test_search_returns_200(self):
        client = TestClient(app)
        response = client.get("/api/movies/search?q=Inception")
        assert response.status_code == 200

    def test_search_with_special_chars_does_not_crash(self):
        client = TestClient(app)
        response = client.get("/api/movies/search?q=Inception (2010)")
        assert response.status_code == 200
        response = client.get("/api/movies/search?q=Inception (")
        assert response.status_code == 200

    def test_recommend_top_n_bounds(self):
        client = TestClient(app)
        assert client.get("/api/movies/recommend?title=Inception&top_n=0").status_code == 422
        assert client.get("/api/movies/recommend?title=Inception&top_n=100").status_code == 422
        assert client.get("/api/movies/recommend?title=Inception&top_n=6").status_code == 200

    def test_tmdb_fallback_without_key(self, monkeypatch):
        monkeypatch.setenv("TMDB_API_KEY", "")
        monkeypatch.setattr("backend.main.TMDB_API_KEY", "")
        client = TestClient(app)
        response = client.get("/api/movies/details?title=Inception&year=2010")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "placeholder"
        assert data["poster_url"] is None
        assert "TMDB API key is not configured" in data["overview"]


# ─── Enrichment async tests ───────────────────────────────────────────────────

class TestMovieEnrichmentAsync:
    def test_returns_three_tuple(self, mock_movies_df):
        """get_movie_enrichment_async must return (rating, genres_str, keywords_str)."""
        import httpx

        async def run():
            async with httpx.AsyncClient() as client:
                with patch.dict("os.environ", {"TMDB_API_KEY": ""}):
                    result = await get_movie_enrichment_async(client, "Inception", 2010, mock_movies_df)
            return result

        rating, genres, keywords = asyncio.run(run())
        assert isinstance(rating, float)
        assert isinstance(genres, str)
        assert isinstance(keywords, str)

    def test_genres_not_merged_with_keywords(self, mock_movies_df):
        """genres_str must not contain keyword-only terms; keywords must not contain genre names."""
        import httpx

        async def run():
            async with httpx.AsyncClient() as client:
                with patch.dict("os.environ", {"TMDB_API_KEY": ""}):
                    return await get_movie_enrichment_async(client, "Inception", 2010, mock_movies_df)

        _, genres, keywords = asyncio.run(run())
        # From mock_movies_df: Genres="Action, Science Fiction, Adventure", Keywords="dream, virtual reality..."
        assert "dream" not in genres.lower()         # keyword, not a genre
        assert "science fiction" in genres.lower()   # actual genre
        assert "action" in genres.lower()

    def test_empty_title_returns_empty_strings(self, mock_movies_df):
        import httpx

        async def run():
            async with httpx.AsyncClient() as client:
                with patch.dict("os.environ", {"TMDB_API_KEY": ""}):
                    return await get_movie_enrichment_async(client, "NonExistentMovie XYZ", None, mock_movies_df)

        rating, genres, keywords = asyncio.run(run())
        assert rating == 0.0
        assert genres == ""
        assert keywords == ""

    def test_mocked_tmdb_async_uses_csv_genres(self, monkeypatch, mock_movies_df):
        """Even when TMDB returns data, genres/keywords must come from the CSV, not TMDB."""
        monkeypatch.setenv("TMDB_API_KEY", "MOCK_KEY_123")

        class MockResponse:
            def __init__(self, json_data, status_code=200):
                self._json_data = json_data
                self.status_code = status_code
            def json(self): return self._json_data
            def raise_for_status(self): pass

        async def mock_get(self_instance, url, *args, **kwargs):
            if "search/movie" in url:
                return MockResponse({"results": [{"id": 42, "title": "Inception", "original_title": "Inception"}]})
            elif "movie/42/keywords" in url:
                return MockResponse({"keywords": [{"name": "completely different keyword"}]})
            elif "movie/42" in url:
                return MockResponse({"vote_average": 9.5, "genres": [{"name": "Horror"}]})
            return MockResponse({}, 404)

        import httpx
        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        async def run():
            async with httpx.AsyncClient() as client:
                return await get_movie_enrichment_async(client, "Inception", 2010, mock_movies_df)

        rating, genres, keywords = asyncio.run(run())
        assert rating == 9.5                                  # TMDB rating IS used
        assert "Horror" not in genres                         # TMDB genres NOT used — CSV is used
        assert "Action" in genres or "action" in genres.lower()  # CSV genres ARE returned
        assert "completely different keyword" not in keywords      # TMDB keywords NOT used
