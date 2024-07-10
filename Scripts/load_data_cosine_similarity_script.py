


import os
import pandas as pd
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


def load_data_cosine_similarity():
    ### cosine similarity prep ###
    route_orders = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/df_final_version.csv'
    orders = pd.read_csv(route_orders)
    orders['vector_words'] = orders['nombre_producto'] + ' ' + orders['pasillo'] + ' ' + orders['departamento']

    #### Embeddings dataframe ####
    route_embeddings = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Embeddings/orders_embeddings.csv'
    embeddings = pd.read_csv(route_embeddings)

    #### Embeddings numpy array ####
    route_embeddings_matrix = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Embeddings/embeddings_matrix.npy'
    embeddings_matrix = np.load(route_embeddings_matrix)

    return orders, embeddings, embeddings_matrix