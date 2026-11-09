"""Task 4-8: Euclidean and Pearson user similarity for Project 7.

The input ``ratings`` matrix must contain users who rated all 20 popular
movies (no zero entries), as produced by Coder 1. ``trial_user`` is the
20-element rating vector from the MATLAB data file.
"""

from __future__ import annotations

import numpy as np


def _as_rating_matrix(ratings: np.ndarray) -> np.ndarray:
    """Return ratings as a finite 2-D float array with 20 columns."""
    matrix = np.asarray(ratings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("ratings must be a 2-D matrix (users x movies).")
    if matrix.shape[1] != 20:
        raise ValueError(f"ratings must have 20 movie columns; got {matrix.shape[1]}.")
    if matrix.shape[0] == 0:
        raise ValueError("ratings must contain at least one complete user.")
    if not np.isfinite(matrix).all():
        raise ValueError("ratings must contain only finite values.")
    return matrix


def _as_trial_user(trial_user: np.ndarray, movie_count: int = 20) -> np.ndarray:
    """Return the trial user's ratings as a validated one-dimensional vector."""
    vector = np.asarray(trial_user, dtype=float).reshape(-1)
    if vector.size != movie_count:
        raise ValueError(f"trial_user must contain {movie_count} ratings; got {vector.size}.")
    if not np.isfinite(vector).all():
        raise ValueError("trial_user must contain only finite ratings.")
    return vector


def euclidean_distances(ratings: np.ndarray, trial_user: np.ndarray) -> np.ndarray:
    """Compute the L2 distance from ``trial_user`` to each row of ``ratings``.

    This is the vectorized equivalent of ``norm(ratings(i,:) - trial_user)``
    in Task 4. The result ``eucl[i]`` corresponds to ``ratings[i, :]``.
    """
    matrix = _as_rating_matrix(ratings)
    trial = _as_trial_user(trial_user, matrix.shape[1])
    return np.linalg.norm(matrix - trial, ord=2, axis=1)


def centered_ratings(
    ratings: np.ndarray, trial_user: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center each user's ratings around their own mean (Task 6).

    Returns ``(ratings_cent, trial_user_cent)``. The row means are kept as a
    column during subtraction so broadcasting centers each user independently.
    """
    matrix = _as_rating_matrix(ratings)
    trial = _as_trial_user(trial_user, matrix.shape[1])
    ratings_cent = matrix - np.mean(matrix, axis=1, keepdims=True)
    trial_user_cent = trial - np.mean(trial)
    return ratings_cent, trial_user_cent


def pearson_correlations(
    ratings: np.ndarray, trial_user: np.ndarray
) -> np.ndarray:
    """Compute Pearson correlation between the trial user and every row.

    A constant rating vector has no defined Pearson coefficient. To keep
    sorting deterministic and avoid NaN values, such rows receive correlation
    0.0 (no measured linear similarity).
    """
    ratings_cent, trial_user_cent = centered_ratings(ratings, trial_user)
    row_norms = np.linalg.norm(ratings_cent, axis=1)
    trial_norm = np.linalg.norm(trial_user_cent)
    denominators = row_norms * trial_norm
    numerators = ratings_cent @ trial_user_cent
    correlations = np.zeros_like(numerators, dtype=float)
    np.divide(numerators, denominators, out=correlations, where=denominators > 0)
    # Protect against tiny floating-point excursions outside the mathematical range.
    return np.clip(correlations, -1.0, 1.0)


def compare_users(
    ratings: np.ndarray, trial_user: np.ndarray
) -> dict[str, np.ndarray | int | float]:
    """Run Tasks 4-8 and return distances, correlations, and sorted neighbors.

    ``DistIndex`` is sorted ascending (closest Euclidean neighbor first).
    ``PearsonIndex`` is sorted descending (largest correlation first).
    Indices are zero-based row indices into ``ratings``, ready for NumPy.
    """
    matrix = _as_rating_matrix(ratings)
    trial = _as_trial_user(trial_user, matrix.shape[1])

    # Task 4: Euclidean distance to every complete user's 20-vector.
    eucl = np.linalg.norm(matrix - trial, ord=2, axis=1)

    # Task 5: sort distances ascending and select the closest row.
    DistIndex = np.argsort(eucl, kind="stable")
    closest_user_Dist = int(DistIndex[0])
    MinDist = float(eucl[closest_user_Dist])

    # Task 6: center each row and the trial user around their own means.
    ratings_cent, trial_user_cent = centered_ratings(matrix, trial)

    # Task 7: Pearson correlation via the centered dot product.
    row_norms = np.linalg.norm(ratings_cent, axis=1)
    trial_norm = np.linalg.norm(trial_user_cent)
    denominators = row_norms * trial_norm
    numerators = ratings_cent @ trial_user_cent
    pearson = np.zeros(matrix.shape[0], dtype=float)
    np.divide(numerators, denominators, out=pearson, where=denominators > 0)
    pearson = np.clip(pearson, -1.0, 1.0)

    # Task 8: the greatest coefficient is the most similar Pearson neighbor.
    PearsonIndex = np.argsort(-pearson, kind="stable")
    closest_user_Pearson = int(PearsonIndex[0])
    MaxPearson = float(pearson[closest_user_Pearson])

    return {
        "eucl": eucl,
        "MinDist": MinDist,
        "DistIndex": DistIndex,
        "closest_user_Dist": closest_user_Dist,
        "ratings_cent": ratings_cent,
        "trial_user_cent": trial_user_cent,
        "pearson": pearson,
        "MaxPearson": MaxPearson,
        "PearsonIndex": PearsonIndex,
        "closest_user_Pearson": closest_user_Pearson,
    }


if __name__ == "__main__":
    # Example integration with Coder 1's data-loading/preprocessing step.
    from scipy.io import loadmat

    data = loadmat("users_movies.mat")
    users_movies_sort = np.asarray(data["users_movies_sort"])
    ratings = users_movies_sort[np.all(users_movies_sort != 0, axis=1)]
    result = compare_users(ratings, data["trial_user"])
    print(f"Complete users: {ratings.shape[0]}")
    print(f"Closest Euclidean user (ratings row, 0-based): {result['closest_user_Dist']}")
    print(f"Minimum Euclidean distance: {result['MinDist']:.6f}")
    print(f"Closest Pearson user (ratings row, 0-based): {result['closest_user_Pearson']}")
    print(f"Maximum Pearson correlation: {result['MaxPearson']:.6f}")
