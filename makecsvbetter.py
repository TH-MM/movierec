import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

def get_movie_recommendations(user_id, n_recommendations=10):
    # Fetch data from the database
    ratings = Rating.objects.all().values()
    df = pd.DataFrame(ratings)

    # Create a user-movie matrix
    user_movie_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Compute cosine similarity between users
    user_similarity = 1 - cosine(user_movie_matrix, user_movie_matrix)
    np.fill_diagonal(user_similarity, 0)

    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id].values

    # Compute weighted sum of ratings
    user_similarity_scores = user_similarity[user_id]
    weighted_sum = user_similarity_scores.dot(user_movie_matrix)
    similarity_sum = user_similarity_scores.sum()

    # Compute predicted ratings
    predicted_ratings = weighted_sum / similarity_sum

    # Get the movies the user hasn't rated yet
    unrated_movies = user_movie_matrix.loc[user_id] == 0

    # Get top n recommendations
    recommendations = pd.Series(predicted_ratings, index=user_movie_matrix.columns)
    recommendations = recommendations[unrated_movies].sort_values(ascending=False).head(n_recommendations)

    # Fetch movie details
    recommended_movies = Movie.objects.filter(id__in=recommendations.index)
    return recommended_movies
