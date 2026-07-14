#!/usr/bin/env python3
"""
Precompute and cache sentence embeddings for all movies in the dataset.
Run once; re-run with --force to regenerate even if cache already exists.

Usage:
    python backend/build_embeddings.py
    python backend/build_embeddings.py --force

Output:
    backend/movie_embeddings.npy   — float32 array [N, 384], L2-normalized
    backend/movie_index.pkl        — aligned DataFrame (Title, Year, TMDb_Rating,
                                     vote_count, Genres, Keywords)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.recommender import build_blob  # single source of truth for blob format

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "top50k_movies.csv")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "movie_embeddings.npy")
INDEX_PATH = os.path.join(SCRIPT_DIR, "movie_index.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def show_sample_blobs(df: pd.DataFrame) -> None:
    """Print example blobs for a few specific movies."""
    targets = ["Avatar", "Spider-Man: Into the Spider-Verse"]
    for t in targets:
        match = df[df["Title"].str.lower() == t.lower()]
        if not match.empty:
            blob = build_blob(match.iloc[0])
            print(f"\n  [{t}]\n  {blob[:300]}...")

    # One with empty keywords
    empty_kw = df[df["Keywords"].isna() | (df["Keywords"].str.strip() == "")]
    if not empty_kw.empty:
        row = empty_kw.iloc[0]
        blob = build_blob(row)
        print(f"\n  [Empty-keywords sample: {row['Title']}]\n  {blob[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and cache sentence embeddings for the movie dataset."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if cache files already exist."
    )
    args = parser.parse_args()

    # Skip if cache exists and --force not passed
    if not args.force and os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
        print(f"Cache already exists:\n  {EMBEDDINGS_PATH}\n  {INDEX_PATH}")
        print("Pass --force to regenerate.")
        sys.exit(0)

    # Load dataset
    print(f"Loading dataset from:\n  {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, on_bad_lines="skip")
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} rows | columns: {df.columns.tolist()}")

    # Null stats
    kw_nulls = df["Keywords"].isna().sum()
    print(f"Keywords nulls: {kw_nulls:,} ({kw_nulls / len(df) * 100:.1f}% — these rows omit the Keywords clause)")

    # Show sample blobs before encoding
    print("\n=== Sample blobs ===")
    show_sample_blobs(df)
    print()

    # Build text blobs
    print("Building text blobs...")
    blobs = df.apply(build_blob, axis=1).tolist()
    print(f"Built {len(blobs):,} blobs. Avg length: {sum(len(b) for b in blobs) // len(blobs)} chars")

    # Load sentence-transformer model
    print(f"\nLoading model: {MODEL_NAME}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)

    # Encode with L2 normalization (cosine sim = dot product on normalized vectors)
    print(f"Encoding {len(blobs):,} blobs (batch_size={BATCH_SIZE}) with progress bar...")
    embeddings = model.encode(
        blobs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize so cosine sim = dot product
        convert_to_numpy=True
    )

    print(f"\nEmbeddings shape: {embeddings.shape} (dtype={embeddings.dtype})")
    print(f"Sample L2 norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.6f}")

    # Score distribution sample (self-similarity sanity check)
    sample_scores = embeddings[:500] @ embeddings[0]
    sample_scores_sorted = np.sort(sample_scores)[::-1]
    print(f"\nScore distribution (anchor=row0, top-500 sample):")
    print(f"  p99: {np.percentile(sample_scores, 99):.4f}")
    print(f"  p90: {np.percentile(sample_scores, 90):.4f}")
    print(f"  p75: {np.percentile(sample_scores, 75):.4f}")
    print(f"  p50: {np.percentile(sample_scores, 50):.4f}")
    print(f"  p25: {np.percentile(sample_scores, 25):.4f}")

    # Build aligned index DataFrame
    index_df = df[["Title", "Year", "TMDb_Rating", "vote_count", "Genres", "Keywords"]].copy()
    index_df["Keywords"] = index_df["Keywords"].fillna("")
    index_df = index_df.reset_index(drop=True)

    # Save
    print(f"\nSaving embeddings to:\n  {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Saving index to:\n  {INDEX_PATH}")
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index_df, f)

    size_mb = os.path.getsize(EMBEDDINGS_PATH) / 1024 / 1024
    print(f"\nDone. Embeddings file: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
