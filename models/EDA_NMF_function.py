#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import NMF 
from sklearn.preprocessing import StandardScaler
# %%
# Load the data
ratings = pd.read_csv('/Users/CristaVillatoro/Desktop/OWN-tahini-tensor-encounter-notes /week10/05-Web_App/data/ratings.csv')
movies = pd.read_csv('/Users/CristaVillatoro/Desktop/OWN-tahini-tensor-encounter-notes /week10/05-Web_App/data/movies.csv')
# %%
# Checking shapes
ratings.shape, movies.shape

# %%
# Extracting the year out of the title 
movies['year'] = movies.title.str.extract(r'\(([^\)]+)\)')
movies
# %%
# remove the year from title column
movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '')
#%%
movies['title'] = movies.title.str[:-1]
#%%
type(movies)
#%%
movies.head()
#%%
merged_df = ratings.merge(movies, on='movieId', how='left')
merged_df
# %%

# %%
# Count the number of ratings per movie
num_ratings_per_movie = ratings.groupby('movieId')['rating'].count()
#%%
df_pivot = pd.pivot_table(merged_df, values='rating', index='userId', columns='title')
df_pivot
#%%
df_pivot.isna().sum().sum()
#%%
merged_df.isna().sum().sum()
#%%
# %%
nan_cols85 = [i for i in df_pivot.columns if df_pivot[i].isnull().sum() > 0.95*len(df_pivot)]
df = df_pivot.drop(nan_cols85, axis=1)
#%%
df
#%%
# Filter the movies DataFrame to only include movies with at least 30 ratings
popular_movies = df[df.index.isin(num_ratings_per_movie[num_ratings_per_movie >= 30].index)]
popular_movies
#%%
ratings_imputed_itemmean = popular_movies.fillna(popular_movies.mean())
ratings_imputed_itemmean
# %%
ratings_imputed_itemmean.isna().sum().sum()
#%%
ratings_imputed_itemmean.to_csv('ratings_imputed.csv')
#%%
# extract_users
users = ratings_imputed_itemmean.index.to_list()
users
#%%
# extract items
movies_list = ratings_imputed_itemmean.columns.to_list()
movies_list
#%%
movies_l = popular_movies.columns
movies_l = pd.DataFrame(movies_l)
movies_l
#%%
movies_l.to_csv('movies_list.csv')
#%%
######################### MODEL #################################
# Normalize the data
#%%
nmf_model = NMF(n_components=500, max_iter=1000)

# %%
nmf_model.fit(ratings_imputed_itemmean)
# %%

Q_matrix = nmf_model.components_
Q_matrix

# %%
Q_matrix.shape
# %%
Q = pd.DataFrame(data=Q_matrix, columns=movies_list)
Q
# %%
P_matrix = nmf_model.transform(ratings_imputed_itemmean)
P_matrix
# %%
P_matrix.shape
# %%
P = pd.DataFrame(P_matrix, index=users)
P
# %%
R_hat_matrix = np.dot(P_matrix,Q_matrix)
R_hat_matrix
# %%
R_hat = pd.DataFrame(R_hat_matrix, columns=movies_list, index=users)
R_hat
# %%
nmf_model.reconstruction_err_
# %%
import pickle

with open('nmf_model_final.pkl',mode='wb') as file:
    pickle.dump(nmf_model,file)
#%%

with open('nmf_model_final.pkl','rb') as file:
    loaded_model = pickle.load(file)

#%%
loaded_model
#%%
##################### NEW USER ###################
new_user_query = {"Lord of the Rings, The": 5,
                 "Avatar":2,
                 "Titanic":3.5}
#%%

new_user_dataframe =  pd.DataFrame(new_user_query, columns=movies_list, index=['new_user'])
new_user_dataframe
#%%
# using the same imputation as training data

new_user_dataframe_imputed = new_user_dataframe.fillna(ratings.rating.mean())
new_user_dataframe_imputed
#%%
P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)
P_new_user_matrix
#%%
P_new_user_matrix.shape
#%%
# get as dataframe for a better visualizarion
P_new_user = pd.DataFrame(data=P_new_user_matrix,
                         index = ['new_user'])
#%%
R_hat_new_user_matrix = np.dot(P_new_user, Q_matrix)
R_hat_new_user_matrix
#%%
# get as dataframe for a better visualizarion
R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=nmf_model.feature_names_in_,
                         index = ['new_user'])
R_hat_new_user
#%%
sorted_list = R_hat_new_user.transpose().sort_values(by='new_user', ascending=False).index.to_list()
sorted_list
#%%
rated_movies = list(new_user_query.keys())
rated_movies
#%%
recommended_movies = [movie for movie in sorted_list if movie not in rated_movies]
recommended_movies
#%%
movie1 = "Lord of the Rings, The"
movie2 = "Avatar"
movie3 = "Titanic"

rating1 = 5
rating2 = 2
rating3 = 3.5
query = {movie1: rating1, movie2:rating2, movie3:rating3}

def recommend_nmf(query, model, k):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    recommendations = []
    # 1. candidate generation
    # construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies_list, index=['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(ratings.rating.mean())
   
    # 2. scoring
    
    # calculate the score with the NMF model
    P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)
    # get as dataframe for a better visualizarion
    P_new_user = pd.DataFrame(data=P_new_user_matrix, index = ['new_user'])
    R_hat_new_user_matrix = np.dot(P_new_user, Q_matrix)
    
    # 3. ranking
    
    # filter out movies already seen by the user
    sorted_list = R_hat_new_user.transpose().sort_values(by='new_user', ascending=False).index.to_list()
    rated_movies = list(new_user_query.keys())
    
    # return the top-k highest rated movie ids or titles
    recommended_movies = [movie for movie in sorted_list if movie not in rated_movies][:k]
    
    return recommended_movies

#%%
recommend_nmf(query, nmf_model, 10)

# %%
