import pandas as pd

# Load the main dataset
df_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/df_merged_version.csv'
df = pd.read_csv(df_path)

# Function to get the aisle of a product
def get_aisle_of_product(product_name):
    product_info = df[df['nombre_producto'] == product_name]
    if not product_info.empty:
        aisle_number = product_info.iloc[0]['id_pasillo']
        return aisle_number
    else:
        return None

# Function to load aisle ranking data
def load_aisle_ranking_data(aisle_number):
    aisle_file_path = f'/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Aisle_Rankings/ranking_aisle_{aisle_number}.csv'
    aisle_df = pd.read_csv(aisle_file_path)
    return aisle_df

# Function to display recommendations for a given product
def display_recommendations(product_name):
    aisle_number = get_aisle_of_product(product_name)
    if aisle_number is not None:
        aisle_df = load_aisle_ranking_data(aisle_number)
        top_aisle_products = aisle_df.head(10)['nombre_producto'].tolist()
        top_aisle_products = [product for product in top_aisle_products if product != product_name]
        print(f"Other customers also included these items from aisle {aisle_number} where {product_name} is located:")
        for product in top_aisle_products:
            print(f"- {product}")
    else:
        print(f"Product {product_name} not found in the dataset.")


