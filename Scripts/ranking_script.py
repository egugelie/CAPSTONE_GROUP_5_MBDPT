import pandas as pd

file_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/General_Ranking/ranking_general.csv'

# Function to load the ranking data
def load_ranking_data(file_path):
    ranking_df = pd.read_csv(file_path)
    top_products = ranking_df.head(10)['nombre_producto'].tolist()
    return top_products

# Function to display the top products, excluding those already in the cart
def display_top_products(top_products, cart):
    output = ["Top recommendations from the most sold items in the supermarket:"]
    for product in top_products:
        if product not in cart:
            output.append(f"- {product}")
    return output