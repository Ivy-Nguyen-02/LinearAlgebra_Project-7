"""Tasks 10-13: Recommendation engine for Project 7.

Depends on:
    - Coder 1: data loading + `ratings`, `users_movies_sort`, `index_small_python`
    - Coder 2: `compare_users(ratings, trial_user)` similarity engine
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from task_4_8_similarity import compare_users  # Coder 2's module


# --------------------------------------------------------------------------- #
# Recommendation primitives
# --------------------------------------------------------------------------- #

def liked_movies(
    user_ratings: np.ndarray,
    index_small_python: np.ndarray,
    threshold: int = 5,
) -> list[int]:
    """Return 0-based movie indices from the 20 popular movies that the
    user rated >= `threshold` (Task 10: `liked` vector)."""
    user_ratings = np.asarray(user_ratings).reshape(-1)
    index_small = np.asarray(index_small_python).reshape(-1)
    assert user_ratings.size == index_small.size == 20, \
        f"Expected 20 ratings & 20 indices; got {user_ratings.size} / {index_small.size}"
    return [int(index_small[i]) for i in range(20) if user_ratings[i] >= threshold]


def recommend_movies(
    closest_user_idx: int,
    ratings: np.ndarray,
    users_movies: np.ndarray,
    original_user_indices: np.ndarray | None = None,
    threshold: int = 5,
) -> list[int]:
    """Return 0-based movie indices (out of 3952) rated >= threshold by the
    closest user.
    """
    if original_user_indices is not None:
        user_row = int(original_user_indices[closest_user_idx])
        user_ratings_all = np.asarray(users_movies[user_row, :]).reshape(-1)
    else:
        # Fallback: only the 20 popular movies are visible; map back via index_small
        # Caller must pass index_small_python separately if fallback is used.
        raise ValueError(
            "original_user_indices is required to recommend across all 3952 movies. "
            "Ask Coder 1 to expose np.where(np.all(users_movies_sort != 0, axis=1))[0]."
        )

    return [int(k) for k in range(user_ratings_all.size) if user_ratings_all[k] >= threshold]


# --------------------------------------------------------------------------- #
# Title lookup helper (Coder 1's movies array is 3-level nested)
# --------------------------------------------------------------------------- #

def movie_title(movies: np.ndarray, movie_idx: int) -> str:
    """Extract the title string from Coder 1's nested movies array."""
    entry = movies[movie_idx]
    # Structure: array([array(['Toy Story (1995)'], dtype='<U16')], dtype=object)
    while isinstance(entry, np.ndarray):
        entry = entry[0]
    return str(entry)


def print_movie_list(movies: np.ndarray, indices: Iterable[int], label: str) -> None:
    indices = list(indices)
    print(f"\n{label} ({len(indices)} movies):")
    if not indices:
        print("  (none)")
        return
    for rank, idx in enumerate(indices, start=1):
        print(f"  {rank}. {movie_title(movies, idx)}")


# --------------------------------------------------------------------------- #
# Full per-user pipeline (Tasks 10, 11)
# --------------------------------------------------------------------------- #

def run_pipeline_for_user(
    user_ratings: np.ndarray,
    user_label: str,
    ratings: np.ndarray,
    users_movies: np.ndarray,
    movies: np.ndarray,
    index_small_python: np.ndarray,
    original_user_indices: np.ndarray,
    show: bool = True,
) -> dict:
    """Execute Tasks 4-11 for a single user and return a structured result."""
    user_ratings = np.asarray(user_ratings, dtype=float).reshape(-1)

    # Tasks 4-8: similarity computation (Coder 2)
    sim = compare_users(ratings, user_ratings)
    closest_dist = sim["closest_user_Dist"]
    closest_pearson = sim["closest_user_Pearson"]

    # Task 10: liked + recommendations
    liked = liked_movies(user_ratings, index_small_python)
    rec_dist = recommend_movies(closest_dist, ratings, users_movies, original_user_indices)
    rec_pearson = recommend_movies(closest_pearson, ratings, users_movies, original_user_indices)

    result = {
        "user": user_label,
        "myratings": user_ratings.tolist(),
        "closest_user_Dist": closest_dist,
        "closest_user_Dist_original_row": int(original_user_indices[closest_dist]),
        "MinDist": sim["MinDist"],
        "closest_user_Pearson": closest_pearson,
        "closest_user_Pearson_original_row": int(original_user_indices[closest_pearson]),
        "MaxPearson": sim["MaxPearson"],
        "Q2_same_user": closest_dist == closest_pearson,
        "liked_movie_indices": liked,
        "liked_movie_titles": [movie_title(movies, i) for i in liked],
        "recommend_dist_indices": rec_dist,
        "recommend_dist_titles": [movie_title(movies, i) for i in rec_dist],
        "recommend_Pearson_indices": rec_pearson,
        "recommend_Pearson_titles": [movie_title(movies, i) for i in rec_pearson],
    }

    # Task 11: print to screen
    if show:
        print(f"\n{'=' * 70}")
        print(f"USER: {user_label}")
        print(f"{'=' * 70}")
        print(f"Closest user (Euclidean): row {closest_dist} "
              f"(orig. user {result['closest_user_Dist_original_row']}), dist={sim['MinDist']:.4f}")
        print(f"Closest user (Pearson):   row {closest_pearson} "
              f"(orig. user {result['closest_user_Pearson_original_row']}), r={sim['MaxPearson']:.4f}")
        print(f"Q2 — same user by both criteria? {'YES' if result['Q2_same_user'] else 'NO'}")
        print_movie_list(movies, liked, "Movies you liked (5★ among 20 popular)")
        print_movie_list(movies, rec_dist, "Recommendations (Euclidean)")
        print_movie_list(movies, rec_pearson, "Recommendations (Pearson)")

    return result


# --------------------------------------------------------------------------- #
# Batch pipeline for all 7 members + export (your Day-10 deliverable)
# --------------------------------------------------------------------------- #

def run_batch(
    all_myratings: dict[str, np.ndarray],
    ratings: np.ndarray,
    users_movies: np.ndarray,
    movies: np.ndarray,
    index_small_python: np.ndarray,
    original_user_indices: np.ndarray,
    out_dir: str | Path = "output",
    show: bool = True,
) -> list[dict]:
    """Run the pipeline for every group member and return all results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for label, myratings in all_myratings.items():
        if np.asarray(myratings).reshape(-1).size != 20:
            raise ValueError(f"{label}: myratings must have 20 elements.")
        all_results.append(
            run_pipeline_for_user(
                myratings, label, ratings, users_movies,
                movies, index_small_python, original_user_indices, show=show,
            )
        )

    # --- CSV export (flat, one row per user) ---
    csv_path = out_dir / "recommendations_7members.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user", "Q2_same_user", "MinDist", "MaxPearson",
            "closest_user_Dist_original_row", "closest_user_Pearson_original_row",
            "liked_titles", "recommend_dist_titles", "recommend_Pearson_titles",
        ])
        for r in all_results:
            writer.writerow([
                r["user"], r["Q2_same_user"],
                f"{r['MinDist']:.4f}", f"{r['MaxPearson']:.4f}",
                r["closest_user_Dist_original_row"],
                r["closest_user_Pearson_original_row"],
                " | ".join(r["liked_movie_titles"]),
                " | ".join(r["recommend_dist_titles"]),
                " | ".join(r["recommend_Pearson_titles"]),
            ])
    print(f"\n[export] CSV written → {csv_path}")

    # --- JSON export (full structured data for Writer 3) ---
    json_path = out_dir / "recommendations_7members.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[export] JSON written → {json_path}")

    return all_results
