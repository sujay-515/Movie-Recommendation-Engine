import React, { useState } from "react";
import { Film, AlertTriangle } from "lucide-react";
import "./App.css";

import { CURATED_MOVIES } from "./constants";
import { getRecommendations } from "./utils/api";
import MovieCard from "./components/MovieCard";
import MovieDetailModal from "./components/MovieDetailModal";
import SearchBox from "./components/SearchBox";

function App() {
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Modal tracking state
  const [activeMovie, setActiveMovie] = useState(null);

  // Handler to fetch recommendations
  const handleSearch = async (searchTitle) => {
    if (!searchTitle || !searchTitle.trim()) return;
    setIsLoading(true);
    setError(null);
    setActiveMovie(null); // Close modal if open

    try {
      const data = await getRecommendations(searchTitle, 6);
      setSelectedMovie(data.searched_movie);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedMovie(null);
    setRecommendations([]);
    setError(null);
  };

  return (
    <div className="app-container">
      {/* Sticky Top Navigation Bar */}
      <nav className="top-nav">
        <div className="nav-content">
          <div
            className="logo-link"
            onClick={handleReset}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                handleReset();
              }
            }}
            aria-label="Home / Reset Movie Recommendation Engine"
          >
            <Film className="logo-icon" size={24} />
            <span className="logo-text">Movie Recommendation Engine</span>
          </div>

          <SearchBox onSearch={handleSearch} />
        </div>
      </nav>

      {/* Main Content Layout */}
      {isLoading ? (
        <div className="loading-wrapper">
          <div className="loading-spinner"></div>
          <p className="loading-text">Finding movies with similar storylines...</p>
        </div>
      ) : error ? (
        <div className="loading-wrapper">
          <div className="error-message" style={{ display: "block" }}>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem", fontWeight: "700" }}>
              <AlertTriangle size={18} /> Search Error
            </div>
            {error}
          </div>
        </div>
      ) : (
        <>
          {/* Page Headers */}
          <div className="page-intro">
            <span className="page-title-label">
              {selectedMovie ? "Recommendations" : "Discovery Portal"}
            </span>
            <h2 className="page-title">
              {selectedMovie
                ? `Themed Matches for: ${selectedMovie.title}`
                : "Explore Curated Cinematic Themes"
              }
            </h2>
            <p className="page-subtitle">
              {selectedMovie
                ? `Based on your pick, here are ${recommendations.length} movies you might enjoy.`
                : "Pick a featured movie below, or search for any title above to get personalized recommendations."
              }
            </p>
          </div>

          {/* Restructured Layouts */}
          {selectedMovie ? (
            <div className="recommendations-container">
              {/* 1. Large Horizontal Banner for Searched/Anchor Movie */}
              <MovieCard
                movie={selectedMovie}
                type="banner"
                onClick={setActiveMovie}
                isAnchor={true}
              />

              <h3 className="section-heading">Recommendations</h3>

              {/* 2. Uniform Portrait Grid for Recommendations */}
              <div className="movie-grid">
                {recommendations.slice(0, 6).map((movie, idx) => (
                  <MovieCard
                    key={idx}
                    movie={{
                      Title: movie.Title,
                      Year: movie.Year,
                      IMDb_Rating: movie.IMDb_Rating,
                      Similarity: movie.Similarity,
                    }}
                    type="standard"
                    onClick={setActiveMovie}
                  />
                ))}
              </div>
            </div>
          ) : (
            // Default Discovery Landing Layout (Curated Movies in a Uniform Grid)
            <div className="movie-grid">
              {CURATED_MOVIES.map((movie, idx) => (
                <MovieCard
                  key={idx}
                  movie={movie}
                  type="standard"
                  onClick={setActiveMovie}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* Details Dialog Modal */}
      {activeMovie && (
        <MovieDetailModal
          movie={activeMovie}
          onClose={() => setActiveMovie(null)}
          onFindSimilar={handleSearch}
        />
      )}

      <footer>
        <p>Movie Recommendation Engine &copy; 2026 - by Sujay.</p>
      </footer>
    </div>
  );
}

export default App;
