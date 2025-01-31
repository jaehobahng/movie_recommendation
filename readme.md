# Movie Recommendation System

## Overview
This package implements a versatile movie recommendation system leveraging various recommendation algorithms, including collaborative filtering, genre-based similarity, keyword-based similarity, and Bayesian mean rating methods. The package is designed to provide personalized movie recommendations based on user preferences, movie metadata, and collaborative interactions among users.

## Features
- **User-based Collaborative Filtering**: Recommends movies based on the preferences of similar users.
- **Item-based Collaborative Filtering**: Finds movies similar to those a user has rated highly.
- **Genre-based Recommendations**: Suggests movies with similar genres using cosine similarity.
- **Keyword-based Recommendations**: Leverages NLP models to recommend movies with similar themes or keywords.
- **Top-rated Movies**: Utilizes Bayesian Mean to rank movies by adjusting for rating volume.
- **Trending and Recent Releases**: Identifies currently trending movies or recent releases with high engagement.

## Mathematical Concepts

### 1. Collaborative Filtering
Collaborative Filtering utilizes the preferences of multiple users to generate recommendations:
- **User-Based Filtering**: Computes similarity between users using metrics like cosine similarity or Euclidean distance.
- **Item-Based Filtering**: Computes similarity between items (movies) using the same distance metrics.

**Cosine Similarity**:
\[
\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
\]
Where \(A\) and \(B\) are vector representations of user/movie interactions.

### 2. Genre-Based Recommendations
Genre similarity is computed using one-hot encoding of genres and cosine similarity:
- Construct a binary matrix where each column represents a genre.
- Compute cosine similarity between movies to find genre-based neighbors.

### 3. Bayesian Mean Rating
Bayesian averaging adjusts movie ratings by considering both the average rating and the number of ratings:
\[
\text{Bayesian Mean} = \frac{\sum_{i=1}^{n} R_i + C \cdot \mu}{n + C}
\]
Where:
- \(R_i\) is the rating for movie \(i\).
- \(n\) is the number of ratings for the movie.
- \(\mu\) is the global average rating.
- \(C\) is a constant representing the average number of ratings across all movies.

### 4. Keyword-Based Similarity
This method uses NLP to recommend movies with similar themes:
- Uses Sentence-BERT (SentenceTransformer) to convert keywords into embeddings.
- Computes cosine similarity between these embeddings.

### 5. Time-weighted Bayesian Rating
Adjusts the Bayesian Mean by giving more weight to recent ratings using an exponential decay function:
\[
\text{time\_weight} = e^{\frac{t - t_{min}}{t_{max} - t_{min}}}
\]
Where:
- \(t\) is the timestamp of the rating.
- \(t_{min}\) and \(t_{max}\) are the minimum and maximum timestamps in the dataset.

## Installation
Install the package using pip:
```
pip install movie-recommendation-system
```

## Usage
### Initialization
```python
from movie_recommendation_system import recommendation

# Instantiate the recommendation class
rec = recommendation(user=1, number=5, meta=meta_df, ratings=ratings_df)
```

### Recommendation Methods
- **Get User's Recently Watched Movies**:
```python
rec.user_watched(top_n=5)
```

- **Find Similar Movies**:
```python
rec.find_similar_movies(movie_id=10, X=rec.create_X(), metric='cosine')
```

- **Genre-Based Recommendations**:
```python
cosine_sim, movie_mapper, mapper_movie = rec.genre_matrix()
rec.genre_rec(cosine_sim, movie_mapper, mapper_movie, movie=10)
```

- **Keyword-Based Recommendations**:
```python
keyword_matrix = rec.keyword_similarity(keywords_df)
rec.keyword_rec(keyword_matrix, keywords_df, movie=10)
```

- **Top All-time Movies**:
```python
rec.best_alltime()
```

- **Trending Movies**:
```python
rec.currently_trending(date='2024-01-01')
```

## Dependencies
Ensure the following packages are installed:
- numpy
- pandas
- scipy
- scikit-learn
- sentence-transformers
- matplotlib
- Pillow
- python-dotenv
- requests

## License
This project is licensed under the MIT License.

