# Movie Recommendation Engine
An interactive, full-stack movie recommendation system that suggests films sharing core themes, subgenres, and story keywords using a modern **Sentence Embedding similarity pipeline**.

Originally built as a terminal program for Harvard's CS50P, this project has been upgraded to a professional portfolio showcase featuring a **FastAPI backend** (Python) and a **React + Vite frontend** (JavaScript) with real-time movie posters via **The Movie Database (TMDB) API**.

---

## How Recommendations Work
The engine employs a deep semantic recommendation pipeline built on a static dataset of 49,503 movies:

1. **Typo Tolerance & Fuzzy Search**: When a query is received, the engine checks for exact matches in the database. If not found, it resolves spelling errors using `difflib.get_close_matches` to match the closest movie title.
2. **Dense Sentence Embeddings**: Plot overviews and keywords are encoded into 384-dimensional dense vectors using the `all-MiniLM-L6-v2` transformer model. Cosine similarity is computed via a single vectorized matrix dot product on L2-normalized vectors.
3. **Fractional Genre Overlap**: Instead of a hard filter, candidate scores are adjusted by a fractional genre overlap coefficient:
   $$\text{factor} = 0.82 + 0.38 \times \left(\frac{\text{Shared Genres}}{\text{Anchor Genres}}\right)$$
   This applies a $0.82\times$ penalty to mismatched movies and up to a $1.20\times$ bonus for matching profiles, preventing surface-word coincidences (like matching family rabbit movies with psychological thrillers).
4. **Engagement Weight (Soft Vote Multiplier)**: A logarithmic multiplier based on `vote_count` gently lifts well-known titles and depresses low-engagement mockbusters without hard-filtering niche suggestions.
5. **Decoupled TMDB Enrichment**: To maintain offline-friendly speed, TMDB is treated strictly as an asynchronous enrichment layer. Posters, backdrops, overviews, and live user ratings enhance the display but do not alter the core ranking.

---

## Directory Structure
```
├── backend/
│   ├── main.py             # FastAPI entrypoint, routing, CORS, and Pydantic validation
│   ├── recommender.py      # Core ranking logic, fractional genre scaling, and TMDB helper
│   ├── build_embeddings.py # Precomputes & L2-normalizes sentence embeddings for catalog
│   ├── test_api.py         # Integration tests for FastAPI endpoints
│   ├── test_recommender.py # Unit tests for ranking logic & TMDB offline mock
│   ├── requirements.txt    # Production dependencies (FastAPI, Pandas, HTTPX, sentence-transformers)
│   └── .env.example        # Environment configuration template
├── data/
│   ├── top50k_movies.csv   # Unified movie dataset (49,503 rows)
│   └── data_cleaning.ipynb # Dataset cleaning pipeline: kagglehub download, filtering, and export to top50k_movies.csv
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MovieCard.jsx        # Poster-first movie card (grid & banner variants)
│   │   │   ├── MovieDetailModal.jsx # Accessible detail view modal (Esc to close, ARIA)
│   │   │   └── SearchBox.jsx        # Keyboard-navigable autocomplete search bar
│   │   ├── utils/
│   │   │   └── api.js               # API client using Vite environment variable
│   │   ├── App.jsx                  # Main UI container
│   │   ├── App.css                  # Responsive poster-grid layout & styles
│   │   ├── index.css                # Typography & resets
│   │   └── main.jsx                 # React DOM mount point
│   ├── package.json        # Frontend scripts and dependencies (React 19, Vite 8, Lucide)
│   └── ...
├── requirements-dev.txt    # Local development & testing dependencies (pytest)
└── README.md
```

---

## Installation & Setup

### 1. Backend Setup & Run
Navigate to the root directory, configure the environment, and start the FastAPI dev server:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (development version includes pytest)
pip install -r requirements-dev.txt

# After cloning, run the script to generate the sentence embeddings cache
python backend/build_embeddings.py

# (Optional) Add your TMDB API Key for dynamic posters/overviews
cp backend/.env.example backend/.env
# Open backend/.env and set: TMDB_API_KEY=your_tmdb_api_key_here

# Start the FastAPI server
uvicorn backend.main:app --reload
```
The interactive Swagger API documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup & Run
Open a new terminal window:

```bash
cd frontend
npm install

# Start the Vite development server
npm run dev
```
Navigate to `http://localhost:5173/` in your browser.

---

## Running the Tests
To run the full test suite (covering FastAPI routing, input validation, and embedding similarities):

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests (30 passed test cases)
pytest
```
