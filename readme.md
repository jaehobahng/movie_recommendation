# Movie Recommendation System

## Overview
This package implements a versatile movie recommendation system leveraging various recommendation algorithms, including collaborative filtering, genre-based similarity, keyword-based similarity, and Bayesian mean rating methods. The package is designed to provide personalized movie recommendations based on user preferences, movie metadata, and collaborative interactions among users.

---

## Installation

First, clone the repository and navigate into the project directory:
```bash
$ git clone https://github.com/yourusername/movie-recommender.git
$ cd movie-recommender
```

Install the required packages:
```bash
$ pip install -r requirements.txt
```

Ensure you have an OMDb API key and add it to your environment variables:
```bash
$ export IMAGE_API='your_omdb_api_key'
```

---

## Dataset

To use this recommender system, you need the movie dataset. Download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download) and place it in the following directory:

```bash
/data/kaggle_movies/
```


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
$
\text{cosine}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
$
Where $A$ and $B$ are vector representations of user/movie interactions.

### 2. Genre-Based Recommendations
Genre similarity is computed using one-hot encoding of genres and cosine similarity:
- Construct a binary matrix where each column represents a genre.
- Compute cosine similarity between movies to find genre-based neighbors.

### 3. Bayesian Mean Rating
Bayesian averaging adjusts movie ratings by considering both the average rating and the number of ratings:
$
\text{Bayesian Mean} = \frac{\sum_{i=1}^{n} R_i + C \cdot \mu}{n + C}
$
Where:
- $R_i$ is the rating for movie $i$.
- $n$ is the number of ratings for the movie.
- $\mu$ is the global average rating.
- $C$ is a constant representing the average number of ratings across all movies.

### 4. Keyword-Based Similarity
This method uses NLP to recommend movies with similar themes:
- Uses Sentence-BERT (SentenceTransformer) to convert keywords into embeddings.
- Computes cosine similarity between these embeddings.

### 5. Time-weighted Bayesian Rating
Adjusts the Bayesian Mean by giving more weight to recent ratings using an exponential decay function:
$
\text{time\_weight} = e^{\frac{t - t_{min}}{t_{max} - t_{min}}}
$
Where:
- $t$ is the timestamp of the rating.
- $t_{min}$ and $t_{max}$ are the minimum and maximum timestamps in the dataset.

## Usage

### Example

Please refer [to this notebook](./movie_recommendation.ipynb) for output examples and specific use-cases.


### Read Data
```python
import os
import pandas as pd
from src.recommend.utils.read import read_files

folder_path = "./data/kaggle_movies"
dataframes = read_files(folder_path)

credits = dataframes['credits']
keywords = dataframes['keywords']
links = dataframes['links']
links_small = dataframes['links_small']
movies_metadata = dataframes['movies_metadata']
ratings = dataframes['ratings']
ratings_small = dataframes['ratings_small']

movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')
meta = movies_metadata.merge(links, left_on='id', right_on='tmdbId', how='left')
meta = meta[~meta['id'].isnull()]
```
### Initialize Model
- **Initialize Recommendation for User1 with 20 movies**:
```python
from src.recommend.rec import recommendation

# user 1, 20 recommendations, metadata, ratings data
rec = recommendation(1, 20, meta, ratings)
```

### Recommendation Methods
- **Get User's Recently Watched Movies**:
```python
watched_movies = rec.user_watched(20)
meta[meta['movieId'].isin(watched_movies)]['title']
```

- **Genre-Based Recommendations**:
```python
genre_similarity_matrix, movie_mapper, mapper_movie = rec.genre_matrix()

import random
from src.recommend.utils.image import image_url, show_image

genre_recommendation = []
for i in watched_movies:
    similar_movies = rec.genre_rec(genre_similarity_matrix, movie_mapper, mapper_movie, i)
    genre_recommendation.extend(similar_movies)

random.shuffle(genre_recommendation)

images = image_url(genre_recommendation[:10])
show_image(images)
```

- **Users with Similar Taste Recommendations**:
```python
X = rec.create_X()

similar_recommend = rec.similar_user_movies(X)
bayesian_mean_df = rec.nb_mean(similar_recommend)
movie_list = bayesian_mean_df['imdb_id'].values

images = image_url(movie_list[:10])
show_image(images)
```

- **Keyword-Based Recommendations**:
```python
keyword_similarity_matrix = rec.keyword_similarity(keywords)

keyword_recommendation = []
for i in watched_movies:
    similar_movies = rec.keyword_rec(keyword_similarity_matrix, keywords, i)
    keyword_recommendation.extend(similar_movies)

random.shuffle(keyword_recommendation)

images = image_url(keyword_recommendation[:10])
show_image(images)
```

- **Newly Released Movies**:
```python
test = rec.new_releases('2017-07-01')
images = image_url(test)
show_image(images)
```

- **Top All-time Movies**:
```python
images_list = rec.best_alltime()
images = image_url(images_list)
show_image(images)
```

- **Trending Movies**:
```python
movie_list = rec.currently_trending('2017-07-01')
images = image_url(movie_list)
show_image(images)
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

