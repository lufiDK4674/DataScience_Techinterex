import pandas as pd 
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

moviesDf = pd.DataFrame(pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv"))
ratingsDf = pd.DataFrame(pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv"))

moviesDf.sample(n=5)

ratingsDf.sample(n=5)

Movies= pd.merge(moviesDf[['movieId', 'title', 'genres']], ratingsDf[['userId', 'movieId', 'rating']], on='movieId', how='inner')

Movies.sample(5)

# Check for missing values
print("Missing Values:\n")
print(Movies.isnull().sum())

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

