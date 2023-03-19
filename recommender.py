"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""
# Import libraries
import pandas as pd
import numpy as np
from utils import movies
import pickle
from sklearn.metrics.pairwise import cosine_similarity

############################################### NMF
# Load the model
with open('nmf_model_final.pkl','rb') as file:
    loaded_model = pickle.load(file)


model = loaded_model
ratings_imputed_itemmean = pd.read_csv('ratings_imputed.csv', index_col=0)

# Extract the needed data for NMF
# extract_users
users = ratings_imputed_itemmean.index.to_list()
users
# extract items
movies_list = ratings_imputed_itemmean.columns.to_list()

ratings_list = ratings_imputed_itemmean.index.to_list()


############################################### End of NMF needed data

def recommend_random(k=10):
    return movies['title'].sample(k).to_list()

def recommend_nmf(query, model=loaded_model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    recommendations = []
    # 1. candidate generation
    # construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=movies_list, index=['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
    #print(new_user_dataframe_imputed)
    # 2. scoring
    
    # calculate the score with the NMF model
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    # get as dataframe for a better visualizarion
    P_new_user = pd.DataFrame(data=P_new_user_matrix, index = ['new_user'])
    R_hat_new_user_matrix = np.dot(P_new_user, model.components_)
    # get as dataframe for a better visualizarion
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix, columns=model.feature_names_in_,
                         index = ['new_user']) 
    R_hat_new_user
    
    # 3. ranking
    
    # filter out movies already seen by the user
    sorted_list = R_hat_new_user.transpose().sort_values(by='new_user', ascending=False).index.to_list()
    rated_movies = list(query.keys())
    
    # return the top-k highest rated movie ids or titles
    recommended_movies = [movie for movie in sorted_list if movie not in rated_movies][:k]
    
    return recommended_movies

def recommend_neighborhood(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass


######################################### COSINE SIMILARITY 

def recommend_cs(query, titles, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    user_item = pickle.load(open("user_item_cs.pkl", "rb"))
    new_user_array = np.full(shape=(1,user_item.shape[1]), fill_value=0)
    for i,m in enumerate(titles):    
            try:  # in case user doesnt provide data
                new_user_array[0][query.dic[m]] = ratings[i]
                print('test')
            except:
                continue
    target_user = "new_user"
    user_item.loc['new_user']=new_user_array[0]
    cs=cosine_similarity(user_item)
    cs = pd.DataFrame(cs, index=user_item.index, columns=user_item.index)
    related_users = cs.loc[target_user]
    related_users.pop(target_user)
    user_df=user_item.loc[related_users.index[np.argmax(related_users)]].to_frame('predictions')
    user_df=user_df.loc[~user_df.index.isin(titles)] #remove seen movies
    all=list(user_df.sort_values(by='predictions', ascending=False)[:k].index)
    return all
    