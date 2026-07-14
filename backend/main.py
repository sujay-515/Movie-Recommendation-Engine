import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from backend.recommender import (
    load_movies, find_closest_title, get_recommendations_async, get_movie_enrichment_async
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend.main")

# Load environment variables from .env file inside backend/ directory
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and pre-process dataset once at startup
    try:
        load_movies()
        logger.info("Dataset loaded and cached successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset during startup: {e}")
    yield

app = FastAPI(
    title="Movie Recommendation Engine API",
    description="A FastAPI backend for TF-IDF weighted Cosine Similarity movie recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

# Tighten CORS configuration to development-friendly local origins instead of wildcard * with credentials
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic Response Models
class MovieSearchItem(BaseModel):
    title: str
    year: Optional[int] = None
    rating: Optional[float] = None

class RecommendationItem(BaseModel):
    Title: str
    IMDb_Rating: float
    Year: int
    Similarity: float
    RawSimilarity: float

class SearchedMovie(BaseModel):
    title: str
    year: Optional[int] = None
    rating: Optional[float] = None
    genres: str
    keywords: str

class MovieRecommendResponse(BaseModel):
    searched_movie: SearchedMovie
    recommendations: List[RecommendationItem]

class MovieDetailsResponse(BaseModel):
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: str
    release_date: Optional[str] = None
    source: str

@app.get("/")
def read_root():
    return {"message": "Movie Recommendation Engine API is active. Head to /docs for Swagger UI."}

@app.get("/api/movies/search", response_model=List[MovieSearchItem])
def search_movies(q: str = Query(..., description="Query string to search movie titles")):
    """
    Search for movie titles matching the query string.
    Returns autocomplete suggestions using simple literal substring match and difflib as fallback.
    """
    if not q.strip():
        return []

    try:
        df = load_movies()
    except Exception as e:
        logger.error(f"Error loading movies in search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while loading movie catalog.")

    q_lower = q.lower()
    
    # Literal substring match (regex=False to treat input as literal text)
    matches = df[df["Title"].str.lower().str.contains(q_lower, regex=False, na=False)].head(10)
    
    results = []
    for _, row in matches.iterrows():
        results.append({
            "title": row["Title"],
            "year": int(row["Year"]) if "Year" in df.columns and not pd.isna(row["Year"]) else None,
            "rating": float(row["IMDb_Rating"]) if not pd.isna(row["IMDb_Rating"]) else None
        })

    # If no substring matches, fallback to difflib close match
    if not results:
        closest = find_closest_title(q, df)
        if closest:
            row = df[df["Title"] == closest].iloc[0]
            results.append({
                "title": row["Title"],
                "year": int(row["Year"]) if "Year" in df.columns and not pd.isna(row["Year"]) else None,
                "rating": float(row["IMDb_Rating"]) if not pd.isna(row["IMDb_Rating"]) else None
            })

    return results

@app.get("/api/movies/recommend", response_model=MovieRecommendResponse)
async def recommend_movies(
    title: str = Query(..., description="The title of the movie the user likes"),
    top_n: int = Query(5, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Find recommendations based on the input movie title.
    Automatically resolves typos in titles using find_closest_title.
    Similarity calculation is run deterministically on the static CSV tags.
    """
    try:
        df = load_movies()
    except Exception as e:
        logger.error(f"Error loading movies in recommend: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while loading movie catalog.")

    # Try exact match first
    all_titles_lower = {t.lower(): t for t in df["Title"].values}
    search_title = title.strip().lower()

    resolved_title = None
    if search_title in all_titles_lower:
        resolved_title = all_titles_lower[search_title]
    else:
        resolved_title = find_closest_title(title, df)

    if not resolved_title:
        logger.warning(f"No movie title match found for: '{title}'")
        raise HTTPException(status_code=404, detail="No matching movie title found.")

    target_row = df[df["Title"] == resolved_title].iloc[0]

    # Calculate recommendations using pre-computed sentence embeddings
    recommendations = await get_recommendations_async(resolved_title, df, top_n=top_n)

    # Enrich anchor movie with live TMDB rating and clean CSV Genres/Keywords
    async with httpx.AsyncClient() as client:
        year = int(target_row["Year"]) if "Year" in df.columns and not pd.isna(target_row["Year"]) else None
        rating, genres, keywords = await get_movie_enrichment_async(client, resolved_title, year, df)

    return {
        "searched_movie": {
            "title": resolved_title,
            "year": int(target_row["Year"]) if "Year" in df.columns and not pd.isna(target_row["Year"]) else None,
            "rating": rating,
            "genres": genres,
            "keywords": keywords
        },
        "recommendations": recommendations
    }

@app.get("/api/movies/details", response_model=MovieDetailsResponse)
async def get_movie_details(
    title: str = Query(..., description="Title of the movie"),
    year: int = Query(None, description="Year of release"),
    x_tmdb_key: str = Header(None, alias="X-TMDB-KEY")
):
    """
    Fetch movie metadata (poster, backdrop, overview) from TMDB.
    Supports TMDB_API_KEY from environment or X-TMDB-KEY request header.
    """
    api_key = x_tmdb_key or TMDB_API_KEY
    if not api_key:
        return {
            "poster_url": None,
            "backdrop_url": None,
            "overview": "TMDB API key is not configured on the backend. Add TMDB_API_KEY to your backend .env file or enter it in the web UI settings to enable movie posters.",
            "source": "placeholder"
        }

    query = title.strip()
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": api_key,
        "query": query
    }
    if year:
        params["primary_release_year"] = year

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("results"):
                best_match = data["results"][0]
                poster_path = best_match.get("poster_path")
                backdrop_path = best_match.get("backdrop_path")
                
                return {
                    "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
                    "backdrop_url": f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else None,
                    "overview": best_match.get("overview", "No description available."),
                    "release_date": best_match.get("release_date"),
                    "source": "tmdb"
                }
        except Exception as e:
            logger.error(f"TMDB query error for details '{title}': {e}")
            
    return {
        "poster_url": None,
        "backdrop_url": None,
        "overview": "Failed to fetch details from TMDB or movie not found.",
        "source": "fallback"
    }
