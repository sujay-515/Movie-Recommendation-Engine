import React, { useState, useEffect, useRef } from "react";
import { Search, X } from "lucide-react";
import { searchMovies } from "../utils/api";

export default function SearchBox({ onSearch }) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState(-1);
  const suggestionsRef = useRef(null);

  useEffect(() => {
    if (query.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    const delayDebounce = setTimeout(async () => {
      try {
        const data = await searchMovies(query);
        setSuggestions(data);
      } catch (err) {
        console.error("Error fetching suggestions:", err);
      }
    }, 200);

    return () => clearTimeout(delayDebounce);
  }, [query]);

  const handleSelect = (title) => {
    onSearch(title);
    setQuery("");
    setSuggestions([]);
    setActiveSuggestionIndex(-1);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      if (activeSuggestionIndex >= 0 && suggestions[activeSuggestionIndex]) {
        handleSelect(suggestions[activeSuggestionIndex].title);
      } else if (query.trim()) {
        handleSelect(query);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveSuggestionIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveSuggestionIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Escape") {
      setSuggestions([]);
      setActiveSuggestionIndex(-1);
    }
  };

  return (
    <div className="header-search-wrapper">
      <div 
        className="header-search-box"
        role="combobox"
        aria-expanded={suggestions.length > 0}
        aria-haspopup="listbox"
        aria-controls="suggestions-list-id"
      >
        <Search className="header-search-icon" size={16} />
        <input
          id="search-input-field"
          className="header-search-input"
          type="text"
          placeholder="Search a movie you like..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActiveSuggestionIndex(-1);
          }}
          onKeyDown={handleKeyDown}
          aria-autocomplete="list"
          aria-controls="suggestions-list-id"
          aria-activedescendant={
            activeSuggestionIndex >= 0
              ? `suggestion-item-${activeSuggestionIndex}`
              : undefined
          }
        />
        {query && (
          <button 
            id="clear-query-btn" 
            className="header-clear-btn" 
            onClick={() => {
              setQuery("");
              setSuggestions([]);
            }}
            aria-label="Clear search query"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {suggestions.length > 0 && (
        <ul 
          id="suggestions-list-id"
          className="header-suggestions-list" 
          ref={suggestionsRef}
          role="listbox"
          aria-label="Movie search suggestions"
        >
          {suggestions.map((item, idx) => (
            <li
              key={idx}
              id={`suggestion-item-${idx}`}
              className="header-suggestion-item"
              style={{
                backgroundColor: idx === activeSuggestionIndex ? "var(--surface-1)" : "",
              }}
              onClick={() => handleSelect(item.title)}
              role="option"
              aria-selected={idx === activeSuggestionIndex}
            >
              <span className="header-suggestion-title">{item.title}</span>
              {item.year && <span className="header-suggestion-year">{item.year}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
