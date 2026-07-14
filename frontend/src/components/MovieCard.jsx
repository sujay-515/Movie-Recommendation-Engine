import React, { useState, useEffect } from "react";
import { Film, Star } from "lucide-react";
import { getMovieDetails } from "../utils/api";

const getRatingClass = (r) => {
  if (r >= 8.0) return "rating-badge-high";
  if (r >= 6.0) return "rating-badge-mid";
  return "rating-badge-low";
};

export default function MovieCard({ movie, type, onClick, isAnchor = false }) {
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(false);

  const title = movie.Title || movie.title;
  const year = movie.Year || movie.year;
  const rating = movie.IMDb_Rating || movie.rating;
  const similarity = movie.Similarity;

  useEffect(() => {
    let active = true;
    const fetchDetails = async () => {
      setLoading(true);
      try {
        const data = await getMovieDetails(title, year);
        if (active) {
          setDetails(data);
        }
      } catch (err) {
        console.error("Failed to load details for", title, err);
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchDetails();
    return () => {
      active = false;
    };
  }, [title, year]);

  const percentage = similarity ? Math.round(similarity * 100) : null;
  const bgImage = (type === "large" ? details?.backdrop_url : details?.poster_url) || details?.poster_url;

  const handleKeyDown = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onClick({ ...movie, details });
    }
  };

  // Horizontal Banner card layout for the Searched/Anchor movie
  if (type === "banner") {
    const bannerBg = details?.backdrop_url || details?.poster_url;
    const genresRaw = movie.genres || movie.Genres || "";
    
    return (
      <div 
        className="movie-banner-card"
        onClick={() => onClick({ ...movie, details })}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
        aria-label={`View details for ${title}`}
      >
        {bannerBg && (
          <div 
            className="banner-backdrop-bg"
            style={{ backgroundImage: `url(${bannerBg})` }}
          ></div>
        )}
        <div className="banner-overlay"></div>

        <div className="banner-content">
          <div className="banner-poster">
            {loading ? (
              <div className="banner-poster-placeholder animate-pulse">
                <div className="spinner" style={{ width: "20px", height: "20px" }}></div>
              </div>
            ) : details?.poster_url ? (
              <img src={details.poster_url} alt={title} className="banner-poster-img" />
            ) : (
              <div className="banner-poster-placeholder">
                <Film size={20} style={{ opacity: 0.3 }} />
              </div>
            )}
          </div>

          <div className="banner-details">
            <div className="banner-top-meta">
              <span className="featured-badge search-badge">Your Search</span>
              {rating > 0 && (
                <span className={`rating-badge ${getRatingClass(rating)}`}>
                  <Star size={10} fill="currentColor" /> {rating.toFixed(1)}
                </span>
              )}
            </div>

            <h2 className="banner-title">{title}</h2>

            <div className="banner-metadata">
              <span>{year || "N/A"}</span>
              {details?.release_date && (
                <>
                  <div className="card-metadata-dot"></div>
                  <span>Released: {new Date(details.release_date).toLocaleDateString()}</span>
                </>
              )}
            </div>

            {details?.overview && (
              <p className="banner-description">{details.overview}</p>
            )}

            {genresRaw && (
              <div className="banner-genres">
                {genresRaw.split(",").map((g, idx) => (
                  <span key={idx} className="banner-genre-pill">{g.trim()}</span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Standard/Uniform portrait card layout
  return (
    <div 
      className={`bento-card-${type}`}
      onClick={() => onClick({ ...movie, details })}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-label={`View details for ${title}`}
    >
      <div className="movie-card">
        {loading ? (
          <div className="card-poster-placeholder animate-pulse">
            <div className="spinner" style={{ width: "24px", height: "24px" }}></div>
          </div>
        ) : bgImage ? (
          <div 
            className="card-poster-bg"
            style={{ backgroundImage: `url(${bgImage})` }}
          ></div>
        ) : (
          <div className="card-poster-placeholder">
            <Film size={24} style={{ opacity: 0.15, marginBottom: "0.5rem" }} />
            <div style={{ fontSize: "0.85rem", fontWeight: "700", opacity: 0.3 }}>{title}</div>
          </div>
        )}

        <div className="card-overlay"></div>

        <div className="card-top-badges">
          {isAnchor ? (
            <span className="featured-badge search-badge">Your Search</span>
          ) : percentage ? (
            <span className="match-percentage-badge">{percentage}% Match</span>
          ) : (
            <span className="featured-badge">Featured</span>
          )}
          {rating > 0 && (
            <span className={`rating-badge ${getRatingClass(rating)}`}>
              <Star size={10} fill="currentColor" /> {rating.toFixed(1)}
            </span>
          )}
        </div>

        <div className="card-content">
          <h3 className="card-title">{title}</h3>
          
          {type === "large" && details?.overview && (
            <p className="card-description">{details.overview}</p>
          )}

          <div className="card-metadata">
            <span>{year || "N/A"}</span>
            {type === "large" && details?.release_date && (
              <>
                <div className="card-metadata-dot"></div>
                <span>Released: {new Date(details.release_date).toLocaleDateString()}</span>
              </>
            )}
          </div>

          {movie.Tags && (type === "large" || type === "medium") && (
            <div className="card-tags">
              {movie.Tags.split(",").slice(0, 4).map((tag, idx) => (
                <span key={idx} className="card-tag-pill">{tag.trim()}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
