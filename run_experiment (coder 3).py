"""End-to-end experiment driver for Project 7.
"""

import numpy as np
from scipy.io import loadmat

from recommender import run_batch
from task_4_8_similarity import compare_users


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Fill in your group's ratings here (20 ints each, 1-5)
# Order matches index_small_python / the printed top-20 list:
#   1. Toy Story (1995)            11. American President, The (1995)
#   2. Jumanji (1995)              12. Dracula
#   3. Grumpier Old Men (1995)     13. Balto (1995)
#   4. Waiting to Exhale (1995)    14. Nixon (1995)
#   5. Father of the Bride II      15. Cutthroat Island (1995)
#   6. Heat (1995)                 16. Casino (1995)
#   7. Sabrina (1995)              17. Sense and Sensibility (1995)
#   8. Tom and Huck (1995)         18. Four Rooms (1995)
#   9. Sudden Death (1995)         19. Ace Ventura
#  10. GoldenEye (1995)            20. Money Train (1995)
# ═══════════════════════════════════════════════════════════════════════════

MYRATINGS = {
    "Member 1 (Coder 1)":  [4, 3, 2, 2, 2, 4, 5, 4, 4, 4, 4, 3, 3, 4, 3, 5, 4, 2, 4, 5],
    "Member 2 (Coder 2)":  [4, 3, 4, 4, 4, 5, 4, 5, 4, 3, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4],
    "Member 3 (Coder 3)":  [4, 4, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 1, 3, 3, 2, 3, 4, 3],
    "Member 4 (Coder 4)":  [5, 3, 3, 4, 3, 4, 3, 4, 5, 5, 5, 2, 5, 1, 2, 4, 3, 3, 5, 5],
    "Member 5 (Writer 1)": [3, 2, 3, 1, 2, 4, 1, 1, 2, 2, 1, 2, 2, 1, 1, 4, 3, 2, 1, 2],
    "Member 6 (Writer 2)": [4, 4, 5, 3, 3, 5, 3, 4, 4, 5, 5, 5, 4, 4, 3, 5, 5, 4, 5, 4],
    "Member 7 (Writer 3)": [5, 3, 3, 4, 3, 3, 2, 2, 3, 3, 4, 3, 3, 4, 2, 3, 3, 4, 4, 3],
}


def main() -> None:
    # ── Load data (Coder 1) ──────────────────────────────────────────────
    data = loadmat("users_movies.mat")
    users_movies       = np.asarray(data["users_movies"])
    users_movies_sort  = np.asarray(data["users_movies_sort"])
    movies             = data["movies"]
    index_small_python = np.asarray(data["index_small"]).reshape(-1) - 1
    trial_user         = np.asarray(data["trial_user"]).reshape(-1)

    # ── Filter + recover original row indices (Coder 1's job — see note) ─
    complete_mask = np.all(users_movies_sort != 0, axis=1)
    ratings = users_movies_sort[complete_mask]
    original_user_indices = np.where(complete_mask)[0]

    print(f"Complete users: {ratings.shape[0]} (out of {users_movies.shape[0]})")

    # ── Sanity check: original_user_indices must be < 6040 ───────────────
    assert original_user_indices.max() < users_movies.shape[0], \
        "Row mapping broken: users_movies_sort must preserve users_movies row order."

    # ── BASELINE: run trial_user first (Task 11 reference output) ────────
    print("\n" + "█" * 70)
    print("█  BASELINE RUN — trial_user (from users_movies.mat)")
    print("█" * 70)
    run_pipeline_for_user(
        trial_user, "trial_user", ratings, users_movies,
        movies, index_small_python, original_user_indices,
    )

    # ── BATCH: run all 7 members, export CSV+JSON
    print("\n" + "█" * 70)
    print("█  BATCH RUN — 7 group members")
    print("█" * 70)
    run_batch(
        MYRATINGS, ratings, users_movies, movies,
        index_small_python, original_user_indices,
        out_dir="output", show=True,
    )


if __name__ == "__main__":
    from recommender import run_pipeline_for_user
    main()
