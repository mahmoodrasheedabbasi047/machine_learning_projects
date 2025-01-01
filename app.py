import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movie_index = movies_data[movies_data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    recommend_movies = []
    for i in movies_list:
        recommend_movies.append(movies_data.iloc[i[0]].title)
    
    return recommend_movies
    


st.title("Movie Recommendation System")

data = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies_data = pd.DataFrame(data)

selected_movie = st.selectbox(
    "Select a movie",
    movies_data['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(movie=selected_movie)

    for i in recommendations:
        st.write(i)
