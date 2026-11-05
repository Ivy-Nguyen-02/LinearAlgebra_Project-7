from scipy.io import loadmat
import numpy as np

# Task 1: Uploading data from users_movies.mat:
data = loadmat('users_movies.mat')

# Task 2: Converting figures from MATLAB into Python:
index_small = data['index_small'].flatten()
index_small_python = index_small - 1

print("Index gốc:")
print(index_small)

print("\nIndex sau khi chuyển sang Python:")
print(index_small_python)

# Task 3: Listing 20 most popular movies:
movies = data['movies'].flatten()

def show_top20_movies(movies, index_small_python):
    print("TOP 20 MOST POPULAR MOVIES:\n")
   
    for i, idx in enumerate(index_small_python):
        title = movies[idx][0] if isinstance(movies[idx], np.ndarray) else str(movies[idx])
        print(f"{i+1:2d}. {title}")

show_top20_movies(movies, index_small_python)

# Task 4: Creating the standard ratings' matrix:
users_movies_sort = data['users_movies_sort']

def create_ratings_matrix(users_movies_sort):
    ratings = users_movies_sort
    valid_mask = np.all(users_movies_sort !=0, axis=1)
    return ratings, valid_mask
ratings, valid_mask = create_ratings_matrix(users_movies_sort)

print("\nResults of creating ratings' matrix")
print(f"Initial total number of users: {users_movie_sort.shape[0]}")
print(f"Number of valid users: {ratings.shape[0]}")
print(f"Size of the standard ratings' matrix: {ratings.shape}")
