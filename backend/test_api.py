import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Movie Recommendation Engine API" in response.json()["message"]

def test_search_movies():
    response = client.get("/api/movies/search?q=Inception")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert any(item["title"].lower() == "inception" for item in data)

def test_recommend_movies():
    # Test getting recommendations for a valid movie
    response = client.get("/api/movies/recommend?title=Inception")
    assert response.status_code == 200
    data = response.json()
    assert "searched_movie" in data
    assert "recommendations" in data
    assert data["searched_movie"]["title"] == "Inception"
    assert len(data["recommendations"]) <= 5

def test_recommend_movies_not_found():
    response = client.get("/api/movies/recommend?title=NonExistentMoviexyz123")
    assert response.status_code == 404
    assert "detail" in response.json()

def test_get_movie_details():
    # Test details with no TMDB key configured
    response = client.get("/api/movies/details?title=Inception&year=2010")
    assert response.status_code == 200
    data = response.json()
    assert "poster_url" in data
    assert "overview" in data

def test_recommend_movies_tmdb_refresh():
    response = client.get("/api/movies/recommend?title=Inception")
    assert response.status_code == 200
    data = response.json()
    assert "searched_movie" in data
    assert "rating" in data["searched_movie"]
    # Check that rating is updated to TMDB vote_average (approx 8.3)
    assert data["searched_movie"]["rating"] > 8.0
    # Check that genres and keywords are returned separately
    genres = data["searched_movie"]["genres"]
    keywords = data["searched_movie"]["keywords"]
    assert "science fiction" in genres.lower() or "action" in genres.lower()
    assert "dream" in keywords.lower() or "subconscious" in keywords.lower()

    
    # Check candidates have RawSimilarity
    for rec in data["recommendations"]:
        assert "RawSimilarity" in rec

def test_recommend_movies_tmdb_fallback(monkeypatch):
    # Simulate API failure by setting key to an invalid value
    monkeypatch.setenv("TMDB_API_KEY", "INVALID_KEY_XYZ_123")
    
    # We must clear the cache for Inception so it forces a fresh fetch with the invalid key
    from backend.recommender import _tmdb_cache, load_movies
    cache_key = "inception|2010"
    if cache_key in _tmdb_cache:
        del _tmdb_cache[cache_key]

    # Revert Inception's rating in memory to simulate local fallback starting state
    df = load_movies()
    target_rows = df[df["Title"] == "Inception"]
    if not target_rows.empty:
        idx = target_rows.index[0]
        df.at[idx, "IMDb_Rating"] = 8.364

    response = client.get("/api/movies/recommend?title=Inception")
    # Verify the request still completes successfully (200 OK) using fallbacks
    assert response.status_code == 200
    data = response.json()
    assert "searched_movie" in data
    # Local rating fallback (8.364)
    assert data["searched_movie"]["rating"] == 8.364
