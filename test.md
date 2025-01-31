# Movie Recommender Package

Welcome to the **Movie Recommender** package! This Python package leverages collaborative filtering, content-based filtering, and hybrid methods to provide personalized movie recommendations based on user preferences and movie metadata.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Class and Methods Overview](#class-and-methods-overview)
- [Mathematical Concepts](#mathematical-concepts)
- [Recommendation Strategies](#recommendation-strategies)
- [Examples](#examples)
- [Contributing](#contributing)

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

Ensure that the necessary CSV files (e.g., `movies_metadata.csv`, `ratings.csv`) are properly extracted in this directory before running the recommender system.

---

## Usage

The primary interface for this package is through the `recommendation` class. Here’s a quick example:

```python
from movie_recommender import recommendation
import pandas as pd

# Load your data
meta = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings.csv')

# Initialize recommender
recommender = recommendation(user=1, number=5, meta=meta, ratings=ratings)

# Get recommendations
similar_movies = recommender.find_similar_movies(movie_id=1, X=recommender.create_X())
print("Similar movies:", similar_movies)
```

