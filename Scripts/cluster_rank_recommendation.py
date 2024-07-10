import os
import pandas as pd
import joblib

# Path to the model and feature names
model_path1 = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_models/random_forest_model.joblib'
model_path2 = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_models/feature_names.joblib'

# Load the model and feature names
clf = joblib.load(model_path1)
feature_names = joblib.load(model_path2)

# Function to create a filtered product dataframe
def create_filtered_product_dataframe(product_info, product_list):
    filtered_df = product_info[product_info['nombre_producto'].isin(product_list)]
    
    # Count the number of items per Aisle
    df_counts = filtered_df.groupby('pasillo').size().reset_index(name='counts')
    
    # Create a DataFrame with the desired structure
    aisles = product_info['pasillo'].unique()
    data = {f'department_{aisle}': [0] for aisle in aisles}
    
    # Fill with data
    for _, row in df_counts.iterrows():
        data[f'department_{row["pasillo"]}'] = [row['counts']]
    
    # Convert the dictionary into a DataFrame
    df_final = pd.DataFrame(data)
    
    # Ensure the DataFrame has the required feature columns
    df_final = df_final.reindex(columns=feature_names, fill_value=0)
    
    return df_final

# Read the product information CSV file
product_info_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_models/pruduct_info_clustering.csv'
product_info = pd.read_csv(product_info_path, index_col='id_producto')

# Function to load the ranking for the identified cluster
def load_cluster_ranking(cluster_number):
    ranking_path = f'/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_Rankings/ranking_cluster_{cluster_number}.csv'
    ranking_df = pd.read_csv(ranking_path)
    top_cluster_products = ranking_df['nombre_producto'].tolist()
    return top_cluster_products

def predict_cluster(cart):
    if not cart:
        print("The cart is empty. Add products to the cart to make predictions.")
        return []
    
    # Create the sale DataFrame based on the cart
    sale = create_filtered_product_dataframe(product_info, cart)
    
    # Make predictions
    predictions = clf.predict(sale)
    cluster_number = predictions[0]  # Assuming a single prediction for simplicity
    #print("Cluster predictions:", predictions)
    
    # Load the ranking for the identified cluster
    recommended_products = load_cluster_ranking(cluster_number)
    return [product for product in recommended_products if product not in cart]