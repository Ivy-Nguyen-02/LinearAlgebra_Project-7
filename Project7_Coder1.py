from scipy.io import loadmat
import numpy as np

# Task 1: Load dữ liệu
data = loadmat('users_movies.mat')

# Task 2: Chuyển chỉ số MATLAB sang Python
index_small = data['index_small']
index_small_python = index_small - 1

print("Index gốc:")
print(index_small)

print("\nIndex sau khi chuyển sang Python:")
print(index_small_python)

# Task 3: Hiển thị 20 phim phổ biến nhất
movies = data['movies']

def show_top20_movies(movies):
    print("Top 20 movies:\n")

    for i in range(20):
        print(f"{i+1}. {movies[i][0][0]}")

show_top20_movies(movies)

# Task 4: Tạo ma trận ratings hợp lệ
users_movies_sort = data['users_movies_sort']

def create_ratings_matrix(users_movies_sort):
    ratings = users_