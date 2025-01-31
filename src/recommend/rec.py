import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

class recommendation:
    """
    A movie recommendation system that provides various recommendation strategies
    based on user preferences, movie metadata, and collaborative filtering.

    Parameters:
    -----------
    user : int
        The user ID for whom recommendations will be generated.
    number : int
        The number of recommended movies to return.
    meta : pd.DataFrame
        The metadata dataframe containing movie information such as genres, titles, and IMDb IDs.
    ratings : pd.DataFrame
        The ratings dataframe containing user ratings for movies.

    Attributes:
    -----------
    M : int
        Number of unique users in the ratings dataset.
    N : int
        Number of unique movies in the ratings dataset.
    user_mapper : dict
        Dictionary mapping user IDs to indices.
    movie_mapper : dict
        Dictionary mapping movie IDs to indices.
    user_inv_mapper : dict
        Dictionary mapping indices back to user IDs.
    movie_inv_mapper : dict
        Dictionary mapping indices back to movie IDs.
    """

    def __init__(self, user, number, meta, ratings):
        self.user = user
        self.number = number
        self.meta = meta
        self.ratings = ratings

        self.M = self.ratings['userId'].nunique()
        self.N = self.ratings['movieId'].nunique()

        # Map user/movie ID to an index
        self.user_mapper = dict(zip(np.unique(self.ratings["userId"]), list(range(self.M))))
        self.movie_mapper = dict(zip(np.unique(self.ratings["movieId"]), list(range(self.N))))
        
        # Map an index back to a user/movie ID
        self.user_inv_mapper = {v: k for k, v in self.user_mapper.items()}
        self.movie_inv_mapper = {v: k for k, v in self.movie_mapper.items()}

    def user_watched(self, top_n):
        """
        Retrieves the most recently watched movies by the user.

        Parameters:
        -----------
        top_n : int
            The number of most recently watched movies to retrieve.

        Returns:
        --------
        np.ndarray
            A NumPy array containing the IDs of the most recently watched movies.
        """
        temp = self.ratings[self.ratings['userId'] == self.user].sort_values(by='timestamp', ascending=False)[:top_n]
        movie_list = temp['movieId'].values
        return movie_list

    def create_X(self):
        """
        Constructs a sparse user-item matrix from the ratings dataset.

        Returns:
        --------
        csr_matrix
            A sparse matrix representing user-movie interactions, where rows correspond to users
            and columns correspond to movies.
        """
        user_index = [self.user_mapper[i] for i in self.ratings['userId']]
        item_index = [self.movie_mapper[i] for i in self.ratings['movieId']]

        X = csr_matrix((self.ratings["rating"], (user_index, item_index)), shape=(self.M, self.N))
        return X

    def find_similar_movies(self, movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine'):
        """
        Finds movies that are similar to a given movie based on collaborative filtering.

        Parameters:
        -----------
        movie_id : int
            The ID of the movie for which similar movies should be found.
        X : csr_matrix
            The user-item matrix (movies as rows, users as columns).
        movie_mapper : dict
            Dictionary mapping movie IDs to indices.
        movie_inv_mapper : dict
            Dictionary mapping indices back to movie IDs.
        metric : str, optional (default='cosine')
            The distance metric used for similarity calculations.

        Returns:
        --------
        list of int
            A list of movie IDs similar to the given movie.
        """
        X = X.T  # Transpose to make it movie x user
        neighbour_ids = []

        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]

        kNN = NearestNeighbors(n_neighbors=self.number+1, algorithm="brute", metric=metric)
        kNN.fit(X)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)

        for i in range(1, self.number+1):  # Skip the first (itself)
            neighbour_ids.append(movie_inv_mapper[neighbour.item(i)])

        return neighbour_ids

    def genre_matrix(self):
        """
        Creates a genre similarity matrix based on movie genres.

        Returns:
        --------
        tuple
            - A cosine similarity matrix for movies based on genre information.
            - A dictionary mapping indices to movie IDs.
            - A dictionary mapping movie IDs to indices.
        """
        try:
            self.meta["genres"] = self.meta["genres"].apply(ast.literal_eval)
        except:
            pass

        self.meta["genres_list"] = self.meta["genres"].apply(lambda x: [d["name"] for d in x] if isinstance(x, list) else [])
        genres = set(g for G in self.meta["genres_list"] for g in G)

        df = self.meta[['movieId', 'genres_list']]
        for g in genres:
            df[g] = df.genres_list.transform(lambda x: int(g in x))

        df = df.reset_index(drop=True)
        movie_mapper = dict(zip(range(df.shape[0]), df["movieId"]))
        mapper_movie = dict(zip(df["movieId"], range(df.shape[0])))
        movie_genres = df.drop(columns=['genres_list', 'movieId'])

        cosine_sim = cosine_similarity(movie_genres, movie_genres)
        return cosine_sim, movie_mapper, mapper_movie

    def genre_rec(self, matrix, movie_mapper, mapper_movie, movie):
        """
        Finds movies similar to a given movie based on genre similarity.

        Parameters:
        -----------
        matrix : np.ndarray
            A precomputed genre similarity matrix.
        movie_mapper : dict
            Dictionary mapping indices to movie IDs.
        mapper_movie : dict
            Dictionary mapping movie IDs to indices.
        movie : int
            The movie ID for which recommendations should be generated.

        Returns:
        --------
        np.ndarray
            A list of similar movie IMDb IDs.
        """
        idx = mapper_movie[movie]
        sim_scores = sorted(list(enumerate(matrix[idx])), key=lambda x: x[1], reverse=True)
        similar_movies = [movie_mapper[i[0]] for i in sim_scores[1:self.number+1]]
        movie_list = self.meta[self.meta['movieId'].isin(similar_movies)]['imdb_id'].values
        return movie_list

    def best_alltime(self):
        """
        Retrieves the highest-rated movies of all time using a Bayesian Mean approach.

        Returns:
        --------
        np.ndarray
            A list of IMDb IDs of the highest-rated movies.
        """
        global_mean = self.ratings["rating"].mean()
        movie_counts = self.ratings.groupby("movieId")["rating"].count()
        movie_rating_sums = self.ratings.groupby("movieId")["rating"].sum()
        C = movie_counts.mean()

        bayesian_mean_ratings = (movie_rating_sums + global_mean * C) / (movie_counts + C)
        bayesian_mean_df = bayesian_mean_ratings.reset_index().rename(columns={"rating": "bayesian_mean"})
        top_movies = bayesian_mean_df.sort_values(by='bayesian_mean', ascending=False).head(self.number)
        top_movies = top_movies.merge(self.meta[['movieId', 'imdb_id']], on='movieId', how='left')

        return top_movies['imdb_id'].values
