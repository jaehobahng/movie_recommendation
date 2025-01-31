from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class recommendation:
    def __init__(self, user, number, meta, ratings):
        self.user = user
        self.number = number
        self.meta = meta
        self.ratings = ratings

        self.M = self.ratings['userId'].nunique()
        self.N = self.ratings['movieId'].nunique()

        # Map user/movie ID to an index (id, index)
        self.user_mapper = dict(zip(np.unique(self.ratings["userId"]), list(range(self.M))))
        self.movie_mapper = dict(zip(np.unique(self.ratings["movieId"]), list(range(self.N))))
        
        # Map an index to a user/movie ID (index,id)
        self.user_inv_mapper = dict(zip(list(range(self.M)), np.unique(self.ratings["userId"])))
        self.movie_inv_mapper = dict(zip(list(range(self.N)), np.unique(self.ratings["movieId"])))


    def user_watched(self, top_n):
        temp = self.ratings[(self.ratings['userId'] == self.user)].sort_values(by='timestamp',ascending=False)[:top_n]
        movie_list = temp['movieId'].values
        return movie_list

    def create_X(self):
        """
        Generates a sparse matrix from ratings dataframe.
        
        Args:
            ratings: pandas dataframe containing 3 columns (userId, movieId, rating)
        
        Returns:
            X: sparse matrix
            user_mapper: dict that maps user id's to user indices
            user_inv_mapper: dict that maps user indices to user id's
            movie_mapper: dict that maps movie id's to movie indices
            movie_inv_mapper: dict that maps movie indices to movie id's
        """
        
        # list of unique indexes for users and movies
        user_index = [self.user_mapper[i] for i in self.ratings['userId']]
        item_index = [self.movie_mapper[i] for i in self.ratings['movieId']]

        
        X = csr_matrix((self.ratings["rating"], (user_index,item_index)), shape=(self.M,self.N))
        
        return X
    
    def find_similar_movies(self, movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine'):
        """
        Finds k-nearest neighbours for a given movie id.
        
        Args:
            movie_id: id of the movie of interest
            X: user-item utility matrix
            k: number of similar movies to retrieve
            metric: distance metric for kNN calculations
        
        Output: returns list of k similar movie ID's
        """
        # Now it is movie x user by transposing it.
        X = X.T
        neighbour_ids = []
        
        movie_ind = movie_mapper[movie_id]
        movie_vec = X[movie_ind]
        if isinstance(movie_vec, (np.ndarray)):
            movie_vec = movie_vec.reshape(1,-1)
        # use k+1 since kNN output includes the movieId of interest
        kNN = NearestNeighbors(n_neighbors=self.number+1, algorithm="brute", metric=metric)
        kNN.fit(X)
        neighbour = kNN.kneighbors(movie_vec, return_distance=False)
        for i in range(0,self.number):
            n = neighbour.item(i)
            neighbour_ids.append(movie_inv_mapper[n])
        neighbour_ids.pop(0)
        return neighbour_ids

    def genre_matrix(self):
        # Convert string representation of list to actual list
        try:
            self.meta["genres"] = self.meta["genres"].apply(ast.literal_eval)
        except:
            pass

        # Extract only the 'name' values
        self.meta["genres_list"] = self.meta["genres"].apply(lambda x: [d["name"] for d in x] if isinstance(x, list) else [])

        genres = set(g for G in self.meta["genres_list"] for g in G)
        df = self.meta[['movieId','genres_list']]

        for g in genres:
            df[g] = df.genres_list.transform(lambda x: int(g in x))

        df = df.reset_index(drop=True)
        movie_mapper = dict(zip(list(range(df.shape[0])), df["movieId"]))
        mapper_movie = dict(zip(df["movieId"], list(range(df.shape[0]))))
        movie_genres = df.drop(columns=['genres_list','movieId'])

        cosine_sim = cosine_similarity(movie_genres, movie_genres)

        return cosine_sim, movie_mapper, mapper_movie


    def genre_rec(self, matrix, movie_mapper, mapper_movie, movie):

        idx = mapper_movie[movie]

        sim_scores = list(enumerate(matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(self.number+1)]
        similar_movies = [movie_mapper[i[0]] for i in sim_scores]

        movie_list = self.meta[self.meta['movieId'].isin(similar_movies)]['imdb_id'].values

        return movie_list
    
    def similar_user_movies(self, user_movie_matrix):

        i = self.user

        similar_users = self.find_similar_movies(i, user_movie_matrix.T, self.user_mapper, self.user_inv_mapper, metric='cosine')

        original_user_list = self.ratings[self.ratings['userId'] == 1]['movieId'].unique()

        similar_recommend = self.ratings[(self.ratings['userId'].isin(similar_users)) & (~self.ratings['movieId'].isin(original_user_list))]

        return similar_recommend

    def nb_mean(self, ratings):
        global_mean = ratings["rating"].mean()

        # Number of ratings per movie (n)
        movie_counts = ratings.groupby("movieId")["rating"].count()

        # Total rating sum per movie
        movie_rating_sums = ratings.groupby("movieId")["rating"].sum()

        # C: Average number of ratings per movie
        C = movie_counts.mean()

        # Compute Bayesian Mean for each movie
        bayesian_mean_ratings = (movie_rating_sums + global_mean * C) / (movie_counts + C)

        # Store results
        bayesian_mean_df = bayesian_mean_ratings.reset_index().rename(columns={"rating": "bayesian_mean"})

        bayesian_mean_df = bayesian_mean_df.merge(self.meta[['movieId','title','imdb_id']], on='movieId',how='left')

        bayesian_mean_df = bayesian_mean_df.sort_values(by='bayesian_mean', ascending= False)

        return bayesian_mean_df

    def new_releases(self, date):
        # temp = merged_df[pd.to_datetime(merged_df['release_date']) >= pd.to_datetime('2017-07-01')][['movieId','release_date']].sort_values(by='release_date',ascending=False)
        release_ratings = self.ratings.merge(self.meta[['movieId','release_date','imdb_id']], on='movieId')
        recent_release = release_ratings[pd.to_datetime(release_ratings['release_date']) >= pd.to_datetime(date)]
        recent_release["time_weight"] = (recent_release["timestamp"] - recent_release["timestamp"].min()) / (recent_release["timestamp"].max() - recent_release["timestamp"].min())

        # Apply an exponential decay function (e.g., exp(weight) to give more importance to recent_release recent_release)
        recent_release["time_weight"] = np.exp(recent_release["time_weight"])
        # Weighted sum of ratings per movie
        weighted_rating_sums = recent_release.groupby("imdb_id").apply(lambda x: np.sum(x["rating"] * x["time_weight"]))

        # Weighted count (sum of weights per movie)
        weighted_counts = recent_release.groupby("imdb_id")["time_weight"].sum()

        # Adjusted global mean with weighted ratings
        weighted_global_mean = np.sum(recent_release["rating"] * recent_release["time_weight"]) / np.sum(recent_release["time_weight"])


        # Number of ratings per movie (n)
        movie_counts = recent_release.groupby("imdb_id")["rating"].count()

        C = movie_counts.mean()

        # Compute Bayesian Mean with time-adjusted weighting
        time_weighted_bayesian_mean = (weighted_rating_sums + weighted_global_mean * C) / (weighted_counts + C)

        # Store results
        time_weighted_bayesian_recent_release = time_weighted_bayesian_mean.reset_index().rename(columns={0: "time_weighted_bayesian_mean"})

        movie_list = list(time_weighted_bayesian_recent_release.sort_values(by='time_weighted_bayesian_mean', ascending=False)['imdb_id'].head(self.number))

        return movie_list

    def currently_trending(self, date):
        
        self.ratings["date"] = pd.to_datetime(self.ratings["timestamp"], unit="s").dt.date
        recent = self.ratings[self.ratings['date'] >= pd.Timestamp(date).date()]
        recent["time_weight"] = (recent["timestamp"] - recent["timestamp"].min()) / (recent["timestamp"].max() - recent["timestamp"].min())
        # Apply an exponential decay function (e.g., exp(weight) to give more importance to recent recent)
        recent["time_weight"] = np.exp(recent["time_weight"])

        # Weighted sum of ratings per movie
        weighted_rating_sums = recent.groupby("movieId").apply(lambda x: np.sum(x["rating"] * x["time_weight"]))

        # Weighted count (sum of weights per movie)
        weighted_counts = recent.groupby("movieId")["time_weight"].sum()

        # Adjusted global mean with weighted ratings
        weighted_global_mean = np.sum(recent["rating"] * recent["time_weight"]) / np.sum(recent["time_weight"])


        # Number of ratings per movie (n)
        movie_counts = recent.groupby("movieId")["rating"].count()

        C = movie_counts.mean()

        # Compute Bayesian Mean with time-adjusted weighting
        time_weighted_bayesian_mean = (weighted_rating_sums + weighted_global_mean * C) / (weighted_counts + C)

        # Store results
        time_weighted_bayesian_recent = time_weighted_bayesian_mean.reset_index().rename(columns={0: "time_weighted_bayesian_mean"})

        a = time_weighted_bayesian_recent.sort_values(by='time_weighted_bayesian_mean', ascending=False).head(self.number)
        a = a.merge(self.meta[['movieId','imdb_id']],  on='movieId',how='left')
        movie_list = a['imdb_id'].values 

        return movie_list

    def best_alltime(self):

        # Global mean rating (m)
        global_mean = self.ratings["rating"].mean()

        # Number of ratings per movie (n)
        movie_counts = self.ratings.groupby("movieId")["rating"].count()

        # Total rating sum per movie
        movie_rating_sums = self.ratings.groupby("movieId")["rating"].sum()

        # C: Average number of ratings per movie
        C = movie_counts.mean()

        # Compute Bayesian Mean for each movie
        bayesian_mean_ratings = (movie_rating_sums + global_mean * C) / (movie_counts + C)

        # Store results
        bayesian_mean_df = bayesian_mean_ratings.reset_index().rename(columns={"rating": "bayesian_mean"})

        a = bayesian_mean_df.sort_values(by='bayesian_mean', ascending=False).head(self.number)
        a = a.merge(self.meta[['movieId','imdb_id']],  on='movieId',how='left')
        movie_list = a['imdb_id'].values 

        return movie_list

    def keyword_similarity(self, keywords):
        # Load a fast sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast!

        # Convert string representation of list to actual list
        try:
            keywords["keywords"] = keywords["keywords"].apply(ast.literal_eval)
        except:
            pass

        # Extract only the 'name' values
        keywords["keywords_list"] = keywords["keywords"].apply(lambda x: [d["name"] for d in x] if isinstance(x, list) else [])

        keywords['keywords_list'] = keywords['keywords_list'].apply(lambda x: ' '.join(x))

        # Encode movie keywords into embeddings
        keywords['vector'] = keywords['keywords_list'].apply(lambda x: model.encode(x))

        # Stack vectors for similarity computation
        movie_vectors = np.vstack(keywords['vector'].values)

        keyword_similarity_matrix = cosine_similarity(movie_vectors)

        return keyword_similarity_matrix

    def keyword_rec(self, matrix, keywords, movie):

        keywords = keywords.merge(self.meta, left_on='id', right_on='tmdbId', how='left')
        movie_index = dict(zip((keywords["movieId"]), list(range(keywords.shape[0]))))
        index_movie = dict(zip(list(range(keywords.shape[0])), (keywords["movieId"])))

        idx= movie_index[movie]
        similar_movies = list(enumerate(matrix[idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:self.number+1]
        similar_movie_list = [index_movie[i[0]] for i in similar_movies]

        # Fetch images
        movie_list = self.meta[self.meta['movieId'].isin(similar_movie_list)]['imdb_id'].values

        return movie_list