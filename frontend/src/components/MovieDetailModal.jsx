import React, { useEffect } from "react";
import { X, Star, Activity, Film } from "lucide-react";

const getRatingClass = (r) => {
  if (r >= 8.0) return "rating-badge-high";
  if (r >= 6.0) return "rating-badge-mid";
  return "rating-badge-low";
};

export default function MovieDetailModal({ movie, onClose, onFindSimilar }) {
  const title = movie.Title || movie.title;
  const year = movie.Year || movie.year;
  const rating = movie.IMDb_Rating || movie.rating;
  const details = movie.details;

  // New dataset sends genres and keywords as separate fields.
  // Curated landing-page movies (CURATED_MOVIES) still carry a genres field directly.
  const genresRaw = movie.genres || movie.Genres || "";
  const keywordsRaw = movie.keywords || movie.Keywords || "";

  const genreList = genresRaw
    ? genresRaw.split(",").map((g) => g.trim()).filter(Boolean)
    : [];

  // Cap keywords at 12 pills to avoid the modal overflowing
  const keywordList = keywordsRaw
    ? keywordsRaw.split(",").map((k) => k.trim()).filter(Boolean).slice(0, 12)
    : [];

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title-id"
    >
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          className="modal-close-btn"
          onClick={onClose}
          aria-label="Close modal"
        >
          <X size={20} />
        </button>

        {details?.backdrop_url ? (
          <div
            className="modal-header-image"
            style={{ backgroundImage: `url(${details.backdrop_url})` }}
          >
            <div className="modal-backdrop-overlay"></div>
          </div>
        ) : (
          <div className="modal-header-image" style={{ backgroundColor: "var(--surface-1)" }}>
            <div className="modal-backdrop-overlay"></div>
          </div>
        )}

        <div className="modal-body">
          <div className="modal-body-layout">
            <div className="modal-poster-wrapper">
              {details?.poster_url ? (
                <img src={details.poster_url} alt={title} className="modal-poster-img" />
              ) : (
                <div className="card-poster-placeholder">
                  <Film size={24} style={{ opacity: 0.2 }} />
                </div>
              )}
            </div>

            <div className="modal-info-panel">
              <div className="modal-meta-row">
                {year && <span className="header-suggestion-year">{year}</span>}
                {rating > 0 && (
                  <span className={`rating-badge ${getRatingClass(rating)}`}>
                    <Star size={10} fill="currentColor" /> {rating.toFixed(1)} TMDb
                  </span>
                )}
              </div>
              <h2 id="modal-title-id" className="modal-title">{title}</h2>
              <p className="modal-plot">
                {details?.overview || "No synopsis available for this title."}
              </p>

              {/* Genres — distinct pill style */}
              {genreList.length > 0 && (
                <div className="modal-tag-group">
                  <span className="modal-tag-group-label">Genres</span>
                  <div className="modal-tags">
                    {genreList.map((g, idx) => (
                      <span key={idx} className="modal-tag-badge modal-tag-genre">
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Keywords — subtler pill style, capped at 12 */}
              {keywordList.length > 0 && (
                <div className="modal-tag-group">
                  <span className="modal-tag-group-label">Keywords</span>
                  <div className="modal-tags">
                    {keywordList.map((k, idx) => (
                      <span key={idx} className="modal-tag-badge modal-tag-keyword">
                        {k}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="modal-cta-row">
          <button
            className="modal-cta-btn"
            onClick={() => onFindSimilar(title)}
            aria-label={`Find movies similar to ${title}`}
          >
            <Activity size={16} /> Find Similar Movies
          </button>
        </div>
      </div>
    </div>
  );
}
