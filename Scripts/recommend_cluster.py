import joblib
import pandas as pd
from google.colab import drive


# Load the feature names and the random forest model
feature_names_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_models/feature_names.joblib'
model_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_models/random_forest_model.joblib'

feature_names = joblib.load(feature_names_path)
random_forest_model = joblib.load(model_path)

route = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/df_merged_version.csv'
destination = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged'

# Path to the CSV file
file_path = route
# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)




# create product DataFrame
def create_filtered_product_dataframe(pruduct_info, product_list):
    filtered_df = pruduct_info[pruduct_info['nombre_producto'].isin(product_list)]
    df_counts = filtered_df.groupby('pasillo').size().reset_index(name='counts')
    aisles = pruduct_info['pasillo'].unique()
    data = {f'department_{aisle}': [0] for aisle in aisles}
    for _, row in df_counts.iterrows():
        data[f'department_{row["pasillo"]}'] = [row['counts']]
    df_final = pd.DataFrame(data)
    df_final = df_final[feature_names]
    return df_final
    
# Function to get the cluster of the cart
def get_cart_cluster(cart):
    pruduct_info = df[['id_producto', 'nombre_producto', 'id_pasillo','id_departamento','pasillo', 'departamento' ]]
    pruduct_info =  pruduct_info.drop_duplicates(subset=['id_producto'])
    pruduct_info =  pruduct_info.set_index('id_producto')
    features = create_filtered_product_dataframe(pruduct_info, cart)
    cluster = random_forest_model.predict(features)
    return cluster[0]

# Function to load cluster ranking data
def load_cluster_ranking_data(cluster_number):
    cluster_file_path = f'/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Cluster_Rankings/ranking_cluster_{cluster_number}.csv'
    cluster_df = pd.read_csv(cluster_file_path)
    top_cluster_products = cluster_df.head(10)['nombre_producto'].tolist()
    return top_cluster_products

# Function to display cluster recommendations for a given cart
def display_cluster_recommendations(cart):
    cluster = get_cart_cluster(cart)
    top_cluster_products = load_cluster_ranking_data(cluster)
    print(f"Based on your cart, you belong to cluster {cluster}. Here are the top products in this cluster:")
    for product in top_cluster_products:
        if product not in cart:
            print(f"- {product}")
