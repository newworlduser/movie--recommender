import pickle
import streamlit as st
import pandas as pd

import requests

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

# Load movie data and similarity matrix with error handling
try:
    movies = pickle.load(open('movie_list.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))
    
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
    st.info("Make sure 'movie_list.pkl' and 'similarity.pkl' are in the same directory as the app.")
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.info("Please check the console for more details and try again.")
    import traceback
    st.text(traceback.format_exc())





