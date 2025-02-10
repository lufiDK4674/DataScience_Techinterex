# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:18.888975Z","iopub.execute_input":"2025-02-10T11:10:18.889468Z","iopub.status.idle":"2025-02-10T11:10:19.260164Z","shell.execute_reply.started":"2025-02-10T11:10:18.889422Z","shell.execute_reply":"2025-02-10T11:10:19.259383Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:19.261423Z","iopub.execute_input":"2025-02-10T11:10:19.261928Z","iopub.status.idle":"2025-02-10T11:10:42.600092Z","shell.execute_reply.started":"2025-02-10T11:10:19.261894Z","shell.execute_reply":"2025-02-10T11:10:42.599322Z"},"jupyter":{"outputs_hidden":false}}
moviesDf = pd.DataFrame(pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv"))
ratingsDf = pd.DataFrame(pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv"))

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:42.601654Z","iopub.execute_input":"2025-02-10T11:10:42.601902Z","iopub.status.idle":"2025-02-10T11:10:42.630332Z","shell.execute_reply.started":"2025-02-10T11:10:42.601882Z","shell.execute_reply":"2025-02-10T11:10:42.629381Z"},"jupyter":{"outputs_hidden":false}}
moviesDf.sample(n=5)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:42.631780Z","iopub.execute_input":"2025-02-10T11:10:42.632161Z","iopub.status.idle":"2025-02-10T11:10:43.695085Z","shell.execute_reply.started":"2025-02-10T11:10:42.632127Z","shell.execute_reply":"2025-02-10T11:10:43.694157Z"},"jupyter":{"outputs_hidden":false}}
ratingsDf.sample(n=5)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:43.695953Z","iopub.execute_input":"2025-02-10T11:10:43.696249Z","iopub.status.idle":"2025-02-10T11:10:47.186385Z","shell.execute_reply.started":"2025-02-10T11:10:43.696224Z","shell.execute_reply":"2025-02-10T11:10:47.185520Z"},"jupyter":{"outputs_hidden":false}}
Movies= pd.merge(moviesDf[['movieId', 'title', 'genres']], ratingsDf[['userId', 'movieId', 'rating']], on='movieId', how='inner')

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:47.187245Z","iopub.execute_input":"2025-02-10T11:10:47.187492Z","iopub.status.idle":"2025-02-10T11:10:48.234936Z","shell.execute_reply.started":"2025-02-10T11:10:47.187473Z","shell.execute_reply":"2025-02-10T11:10:48.234119Z"},"jupyter":{"outputs_hidden":false}}
Movies.sample(5)

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:48.235837Z","iopub.execute_input":"2025-02-10T11:10:48.236171Z","iopub.status.idle":"2025-02-10T11:10:50.247415Z","shell.execute_reply.started":"2025-02-10T11:10:48.236145Z","shell.execute_reply":"2025-02-10T11:10:50.246291Z"},"jupyter":{"outputs_hidden":false}}
# Check for missing values
print("Missing Values:\n")
print(Movies.isnull().sum())

# %% [code] {"execution":{"iopub.status.busy":"2025-02-10T11:10:50.249532Z","iopub.execute_input":"2025-02-10T11:10:50.249823Z","iopub.status.idle":"2025-02-10T11:19:02.539700Z","shell.execute_reply.started":"2025-02-10T11:10:50.249796Z","shell.execute_reply":"2025-02-10T11:19:02.538529Z"},"jupyter":{"outputs_hidden":false}}
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Define Reader format
reader = Reader(rating_scale=(0.5, 5.0))

# Load dataset
data = Dataset.load_from_df(Movies[['movieId', 'userId', 'rating']], reader)

# Train/Test split
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD model for collaborative filtering
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Check accuracy
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

def recommend_movies(user_id, num_recommendations):
    
    all_movie_ids = Movies['movieId'].unique()
    
    rated_movies = Movies[Movies['userId'] == user_id]['movieId'].values
    unrated_movies = [movie for movie in all_movie_ids if movie not in rated_movies]
    
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_movies = predictions[:num_recommendations]
    recommended_titles = [Movies[Movies['movieId'] == pred.iid]['title'].values[0] for pred in top_movies]
    
    return recommended_titles

# Example Usage: Recommend movies for user 1
userID=input("Enter Your UserID: ")
num_recommendation = int(input("Number of recommendations you want= "))
print(f"Recommended Movies are \n",recommend_movies(userID, num_recommendation))

