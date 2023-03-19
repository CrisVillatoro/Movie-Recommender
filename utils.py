"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np

import tmdbsimple as tmdb

tmdb.API_KEY = '0bf69a28ee0108f01839a18c66bf4d73'

movies = pd.read_csv('movies_list.csv')

def movie_to_id(string_titles):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movies.set_index('title').loc[string_titles]['movieid']
    movieID = movieID.tolist()
    
    return movieID

def id_to_movie(movieID):
    '''
    converts movie Id to title
    '''
    rec_title = movies.set_index('movieid').loc[movieID]['title']
    
    return rec_title

def get_poster_links(movie_list):
    poster_list = []
    for movie in movie_list:
        movie = movie[:10]
        search2 = tmdb.Search()
        response = search2.movie(query=movie)
        if search2.results:
            movie_id = search2.results[0]['id']
            movie = tmdb.Movies(movie_id)
            response = movie.info()
            poster = movie.images()
            poster = poster.get('posters')
            if poster:
                poster = poster[0]
                file_path = poster['file_path']
                file_path = "https://image.tmdb.org/t/p/w500" + file_path
                poster_list.append(file_path)          
        else:
            poster_list.append("https://i.ibb.co/HG65553/hl-PLsovz-Je6j-GKp-QSp31f2-Mx-AMM.png")
    return poster_list