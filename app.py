import pickle
import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

def fetch_poster(movie_id):
    try:
        url = "https://api.themoviedb.org/3/movie/{}?api_key=0d0717e9a071ce2019fc7e6a10502a98&language=en-US".format(movie_id)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Check if poster_path exists
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        else:
            # Return a placeholder image if no poster is available
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    except Exception as e:
        st.error(f"Error fetching poster for movie ID {movie_id}: {str(e)}")
        return "https://via.placeholder.com/500x750?text=Error+Loading+Poster"

def recommend(movie):
    try:
        # Check if the movie exists in our dataset
        if movie not in movies['title'].values:
            st.error(f"Movie '{movie}' not found in the dataset.")
            return [], []
            
        # Get the index of the movie
        index = movies[movies['title'] == movie].index[0]
        
        # Get the similarity scores for this movie with all others
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        
        recommended_movie_names = []
        recommended_movie_posters = []
        
        # Get the top 5 similar movies
        for i in distances[1:6]:
            try:
                # fetch the movie poster
                movie_id = movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(fetch_poster(movie_id))
                recommended_movie_names.append(movies.iloc[i[0]].title)
            except Exception as e:
                st.warning(f"Could not process recommendation: {str(e)}")
                # Add placeholder data if there's an error
                recommended_movie_posters.append("https://via.placeholder.com/500x750?text=Error")
                recommended_movie_names.append("Error loading recommendation")
                
        return recommended_movie_names, recommended_movie_posters
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        # Return empty lists if there's an error
        return [], []


st.header('Movie Recommender System')

# Function to generate pickle files
def generate_pickle_files():
    st.info("Generating recommendation data. This may take a minute...")
    progress_bar = st.progress(0)
    
    try:
        # Define paths to CSV files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        movies_csv = os.path.join(current_dir, 'tmdb_5000_movies.csv')
        credits_csv = os.path.join(current_dir, 'tmdb_5000_credits.csv')
        
        # Check if CSV files exist
        if not os.path.exists(movies_csv) or not os.path.exists(credits_csv):
            st.error("Required CSV files not found. Please make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same directory as the app.")
            return None, None
        
        # Read the CSV files
        movies_df = pd.read_csv(movies_csv)
        credits_df = pd.read_csv(credits_csv)
        progress_bar.progress(10)
        
        # Merge the dataframes
        movies_df = movies_df.merge(credits_df, on='title')
        progress_bar.progress(20)
        
        # Select relevant columns
        movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
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
            L = []
            counter = 0
            for i in ast.literal_eval(text):
                if counter < 3:
                    L.append(i['name'])
                    counter += 1
            return L
        
        # Function to remove spaces from lists
        def collapse(L):
            L1 = []
            for i in L:
                L1.append(i.replace(" ", ""))
            return L1
        
        # Apply the functions to the dataframe
        movies_df['genres'] = movies_df['genres'].apply(convert)
        progress_bar.progress(30)
        movies_df['keywords'] = movies_df['keywords'].apply(convert)
        progress_bar.progress(40)
        movies_df['cast'] = movies_df['cast'].apply(fetch3)
        progress_bar.progress(50)
        movies_df['crew'] = movies_df['crew'].apply(fetch_director)
        progress_bar.progress(60)
        
        # Remove spaces
        movies_df['cast'] = movies_df['cast'].apply(collapse)
        movies_df['crew'] = movies_df['crew'].apply(collapse)
        movies_df['genres'] = movies_df['genres'].apply(collapse)
        movies_df['keywords'] = movies_df['keywords'].apply(collapse)
        
        # Create a new column 'tags' which is a combination of genres, keywords, cast and crew
        movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']
        
        # Create a new dataframe with only the columns we need
        new_df = movies_df[['movie_id', 'title', 'tags']]
        
        # Convert the tags column to a string
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        
        # Convert to lowercase
        new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
        progress_bar.progress(70)
        
        # Create count matrix from the tags
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vector = cv.fit_transform(new_df['tags']).toarray()
        progress_bar.progress(80)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(vector)
        progress_bar.progress(90)
        
        # Save the processed data to pickle files
        pickle.dump(new_df, open('movie_list.pkl', 'wb'))
        pickle.dump(similarity_matrix, open('similarity.pkl', 'wb'))
        progress_bar.progress(100)
        
        st.success("Data generation complete!")
        return new_df, similarity_matrix
    
    except Exception as e:
        st.error(f"Error generating recommendation data: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None, None

# Load movie data and similarity matrix with error handling
try:
    # Check if pickle files exist
    if os.path.exists('movie_list.pkl') and os.path.exists('similarity.pkl'):
        movies = pickle.load(open('movie_list.pkl','rb'))
        similarity = pickle.load(open('similarity.pkl','rb'))
    else:
        # Generate pickle files if they don't exist
        movies, similarity = generate_pickle_files()
        if movies is None or similarity is None:
            st.stop()
    
    # Create the movie selection dropdown
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
    
    # Show recommendations when button is clicked
    if st.button('Show Recommendation'):
        with st.spinner('Getting recommendations...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
            
            # Check if we have recommendations to display
            if recommended_movie_names and recommended_movie_posters:
                # Create 5 columns for the recommendations
                col1, col2, col3, col4, col5 = st.columns(5)
                
                # Display each recommendation in its own column
                if len(recommended_movie_names) >= 1:
                    with col1:
                        st.text(recommended_movie_names[0])
                        st.image(recommended_movie_posters[0])
                        
                if len(recommended_movie_names) >= 2:
                    with col2:
                        st.text(recommended_movie_names[1])
                        st.image(recommended_movie_posters[1])
                
                if len(recommended_movie_names) >= 3:
                    with col3:
                        st.text(recommended_movie_names[2])
                        st.image(recommended_movie_posters[2])
                        
                if len(recommended_movie_names) >= 4:
                    with col4:
                        st.text(recommended_movie_names[3])
                        st.image(recommended_movie_posters[3])
                        
                if len(recommended_movie_names) >= 5:
                    with col5:
                        st.text(recommended_movie_names[4])
                        st.image(recommended_movie_posters[4])
            else:
                st.warning("No recommendations found. Please try another movie.")

except FileNotFoundError as e:
    st.error(f"Error: Could not find required data files. {str(e)}")
    st.info("Attempting to generate recommendation data...")
    # Generate pickle files
    movies, similarity = generate_pickle_files()
    if movies is None or similarity is None:
        st.stop()
    else:
        st.experimental_rerun()
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.info("Please check the console for more details and try again.")
    import traceback
    st.text(traceback.format_exc())





