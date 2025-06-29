import pickle
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the current directory where the script is running
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to CSV files
movies_csv = os.path.join(current_dir, 'tmdb_5000_movies.csv')
credits_csv = os.path.join(current_dir, 'tmdb_5000_credits.csv')

# Define paths for output pickle files
movie_list_pkl = os.path.join(current_dir, 'movie_list.pkl')
similarity_pkl = os.path.join(current_dir, 'similarity.pkl')

print(f"Reading data from:\n{movies_csv}\n{credits_csv}")

# Read the CSV files
try:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)
    print("CSV files loaded successfully.")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Merge the dataframes
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Function to remove spaces from lists
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

# Process the data
print("Processing data...")
try:
    # Convert string representations to lists and extract names
    import ast
    
    # Function to convert string representation of list to actual list
    def convert(text):
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L
    
    # Function to fetch the director name from crew
    def fetch_director(text):
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L
    
    # Function to get the top 3 elements
    def fetch3(text):
        if isinstance(text, list):
            return text[:3]
        L = []
        counter = 0
        for i in ast.literal_eval(text):
            if counter < 3:
                L.append(i['name'])
                counter += 1
        return L
    
    # Apply the functions to the dataframe
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(fetch3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Remove spaces
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    
    # Create a new column 'tags' which is a combination of genres, keywords, cast and crew
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create a new dataframe with only the columns we need
    new = movies[['movie_id', 'title', 'tags']]
    
    # Convert the tags column to a string
    new['tags'] = new['tags'].apply(lambda x: " ".join(x))
    
    # Convert to lowercase
    new['tags'] = new['tags'].apply(lambda x: x.lower())
    
    # Create count matrix from the tags
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(new['tags']).toarray()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vector)
    
    print("Data processing completed successfully.")
except Exception as e:
    print(f"Error processing data: {e}")
    exit(1)

# Save the processed data to pickle files
print(f"Saving pickle files to:\n{movie_list_pkl}\n{similarity_pkl}")
try:
    pickle.dump(new, open(movie_list_pkl, 'wb'))
    pickle.dump(similarity, open(similarity_pkl, 'wb'))
    print("Pickle files saved successfully.")
except Exception as e:
    print(f"Error saving pickle files: {e}")
    exit(1)

print("\nAll done! You can now run the Streamlit app with 'streamlit run app.py'")