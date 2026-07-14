const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const searchMovies = async (query) => {
  if (!query || !query.trim()) return [];
  const res = await fetch(`${API_BASE_URL}/api/movies/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error("Search failed");
  return res.json();
};

export const getRecommendations = async (title, topN = 6) => {
  if (!title || !title.trim()) return null;
  const res = await fetch(`${API_BASE_URL}/api/movies/recommend?title=${encodeURIComponent(title)}&top_n=${topN}`);
  if (!res.ok) {
    const errData = await res.json();
    throw new Error(errData.detail || "Failed to load recommendations.");
  }
  return res.json();
};

export const getMovieDetails = async (title, year) => {
  const res = await fetch(
    `${API_BASE_URL}/api/movies/details?title=${encodeURIComponent(title)}&year=${year || ""}`
  );
  if (!res.ok) throw new Error("Failed to load details");
  return res.json();
};
