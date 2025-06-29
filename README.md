# Movie Recommender System

A content-based movie recommender system using cosine similarity and TMDB dataset.

## Overview

This application recommends movies similar to the one selected by the user. It uses content-based filtering with cosine similarity to find movies with similar characteristics. The system automatically processes the data files when needed, so you don't have to manually generate them.

## Features

- Interactive movie selection from a dropdown menu
- Displays 5 movie recommendations with posters
- Uses TMDB API to fetch movie posters

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Requests
- NumPy
- scikit-learn

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Then select a movie from the dropdown and click "Show Recommendation" to see similar movies.

## Dataset

This project uses the TMDB 5000 Movie Dataset, which includes:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

## Files

- `app.py`: The main Streamlit application that includes automatic data processing
- `movie_list.pkl`: Preprocessed movie data (generated automatically if missing)
- `similarity.pkl`: Precomputed similarity matrix (generated automatically if missing)
- `notebook86c26b4f17.ipynb`: Jupyter notebook with data preprocessing and model creation
- `generate_pickle_files.py`: Script to manually generate the pickle files if needed
