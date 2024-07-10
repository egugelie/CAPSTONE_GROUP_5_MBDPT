
import pandas as pd
import numpy as np
import os
import sys
import imp
from google.colab import drive
from openai import OpenAI
from gtts import gTTS
from IPython.display import Audio
import time
import random
from collections import Counter

script_path = '/content/drive/Shared drives/Capstone/Notebooks/Scripts/'
sys.path.append(script_path)

from voice_recommend_aisle import display_recommendations
from load_data_cosine_similarity_script import load_data_cosine_similarity
from voice_recommender_cosine_similarity_script import recommender_cosine_similarity
from General_Apriori_script import AprioriRecommender
from cluster_rank_recommendation import predict_cluster
from ranking_script import load_ranking_data
from ranking_script import display_top_products

from load_data_cosine_similarity_script import load_data_cosine_similarity

# Load data for cosine similarity
orders, embeddings, embeddings_matrix = load_data_cosine_similarity()


# Random articles as a guide
def random_articles(number):
    # Contar la frecuencia de cada producto
    product_counts = Counter(orders['nombre_producto'])
    # Obtener los 1000 productos más frecuentes
    top_1000_products = [product for product, count in product_counts.most_common(1000)]
    # Seleccionar artículos al azar de los 1000 productos más frecuentes
    random_articles = random.sample(top_1000_products, min(number, len(top_1000_products)))
    
    print("Random articles as a guide:")
    for item in random_articles:
        print(f"- {item}")


#### TECH EXPERIENCE ####


def tech_experience():
    ranking_file_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/General_Ranking/ranking_general.csv'
    top_products = load_ranking_data(ranking_file_path)

    # Function to get user input

    def get_user_input():
      return input("\n Enter the product name to add to your cart (or 0 to finish): ").strip()

    print("\n \n")
    print("Insert the name of the item you included in the cart and press Enter.")
    print("When you are finished, press 0 and Enter to complete your cart.\n")

    print("\n")
    cart = []
    top_products_recommendation = display_top_products(top_products, cart)
    for item in top_products_recommendation:
        print(f"{item}")
    print("\n")

    apriori_recommender = AprioriRecommender('/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Apriori_rules/df_apriori_filtered_rules.csv')

    while True:
        product = get_user_input()
        if product == '0':
            break
        else:          
            cart.append(product)
            print("\n")
            print(f"Added {product} to your cart.\n")

            added_product = "Added," + product + "to your cart."

            #OpenAI prompt
            client = OpenAI(api_key="xxxx") # mati's API key

            response = client.chat.completions.create(
                          model="gpt-3.5-turbo",
                          messages=[
                            {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english simplify)" + added_product}
                          ]
                        )

            #tts = gTTS(f"Added {product} to your cart", lang='es')
            tts = gTTS(text=response.choices[0].message.content, lang='en')
            tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/cart_items.mp3')
            audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/cart_items.mp3'
            # Reproduce el archivo de audio
            display(Audio(audio_path, autoplay=True))


            print("*" * 50)
            print("Current cart contents:")
            for item in cart:
                print(f"- {item}")
            print("*" * 50)
            print("\n")

            if len(cart) <= 2:
              # Call the recommendation function from recommend_aisle.py
              location_recommendation = display_recommendations(product)
              print("\n Other customers included from the same aisle:")
              for item in location_recommendation:
                      print(f"- {item}")

              print("\n")
              # Display the top products again, excluding items in the cart0
              top_products_recommendation = display_top_products(top_products, cart)
              for item in top_products_recommendation:
                      print(f"{item}")


            # Call the Apriori recommendation function if the cart has at least 4 items
            if len(cart) >= 3 and len(cart) < 9:

              # Call the recommendation function from recommend_aisle.py
              location_recommendation = display_recommendations(product)

              # Display the top products again, excluding items in the cart0
              top_products_recommendation = display_top_products(top_products, cart)

              apriori_recommendations = apriori_recommender.get_recommendations(cart)

              # Call the cosine similarity recommendation function for additional recommendations
              cosine_recommendations = recommender_cosine_similarity(cart,orders=orders, embeddings=embeddings, embeddings_matrix=embeddings_matrix)
              cosine_recommendations = cosine_recommendations[:100]
              cosine_recommendations_short = cosine_recommendations[:10]


              print("\n")
              for item in top_products_recommendation:
                      print(f" {item}")
              print("\n")

              print("\n Top recommendations from Location Recommendation:")
              for item in location_recommendation:
                      print(f"- {item}")
              print("\n")

              print("\n Top recommendations from Apriori Algorithm:")
              for item in apriori_recommendations:
                      print(f"- {item}")
              print("\n")

              print("\n Top recommendations from Cosine Algorithm:")
              for item in cosine_recommendations_short:
                      print(f"- {item}")
              print("\n")

              prompt_recommendation = (
                    "The recommendation must start by saying, 'Customers with similar selections to yours also added the following products.' "
                    + "Consider that we are building a product recommendation model for a shopper in a supermarket who currently has the following items in their cart: "
                    + str(cart)
                    + ". Our recommendation models would suggest the following: "
                    + "List of products recommended by Apriori algorithm:"
                    + str(apriori_recommendations)
                    + "List of products recommended by  Cosine similarity algorithm:"
                    + str(cosine_recommendations)
                    + "List of products recommended by Other customers top selections from the aisle that the customer added the last product are:"
                    + str(location_recommendation)
                    + "List of products recommended of Top items sold in the supermarket (excluding the items added in the cart) are:"
                    + str(top_products_recommendation)
                    + "Recommend 5 items (from the list of all the items provided, do not include other items that are not listed) considering complementarity to what is currently in their cart and common sense of what a supermarket cart (shoper behaviour) looks like."
                    + "For the recommendation, prioritize Apriori Algorithm and Cosine similarity."
                    + "List the recommendations but not with number 1,2,3,4,5 (don't forget to start with 'Customers with similar selections to yours also added the following products.')."
                    + "Tag in parenthesis which list did you extract the recommendation from. (Top products, Location Recommendation, Apriori Algorithm or Cosine recommendation)"
                    + "Do not include in the recommendation items that could be the same. Example: if 'Fresas' is already in the cart, do not recommend 'Fresas organicas' as an option."
                    + "If strange items appears in recommendations such as 'REAJUSTE SALARIAL' (that don't seem to be sold in a supermarket), do not include in the recommendations "
                    + "After chosing the 5 items and listing them, don't say nothing more."
                                        )

              #OpenAI prompt
              client = OpenAI(api_key="xxxx") # mati's API key

              response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                              {"role": "assistant", "content": prompt_recommendation}
                            ]
                          )
              openai_response = str(response.choices[0].message.content)

              print("\n")
              print(response.choices[0].message.content)
              print("\n")

              #translate to english the answer for voice spoken recommendation
              response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english , simplify). It must start with:'Customers with similar selections to yours also added the following products.'  Exclude what is included inside the parenthesis in the recommendations (algorithm) " + openai_response}
                            ]
                          )
              tts = gTTS(text=response.choices[0].message.content, lang='en')
              tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_4_more_items.mp3')
              audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_4_more_items.mp3'
              # Reproduce el archivo de audio
              display(Audio(audio_path, autoplay=True))


            # Call the cluster recommendation function if the cart has at least 9 items
            if len(cart) >= 9:

              cluster_recommendations = predict_cluster(cart)

              # Call the cosine similarity recommendation function for additional recommendations
              cosine_recommendations = recommender_cosine_similarity(cart,orders=orders, embeddings=embeddings, embeddings_matrix=embeddings_matrix)
              cosine_recommendations = cosine_recommendations[:100]
              cosine_recommendations_short = cosine_recommendations[:10]


              print("\n")
              for item in top_products_recommendation:
                      print(f" {item}")
              print("\n")

              print("\n Top recommendations from Location Recommendation:")
              for item in location_recommendation:
                      print(f"- {item}")
              print("\n")

              print("\n Top recommendations from Apriori Algorithm:")
              for item in apriori_recommendations:
                      print(f"- {item}")
              print("\n")

              print("\n Top recommendations from Cosine Algorithm:")
              for item in cosine_recommendations_short:
                      print(f"- {item}")
              print("\n")

              print("\n Top recommendations from Cluster Algorithm:")
              for item in cluster_recommendations:
                      print(f"- {item}")
              print("\n")
              

              prompt_recommendation = (
                    "The recommendation must start by saying, 'Customers with similar selections to yours also added the following products.' "
                    + "Consider that we are building a product recommendation model for a shopper in a supermarket who currently has the following items in their cart: "
                    + str(cart)
                    + ". Our recommendation models would suggest the following: "
                    + "List of products recommended by Apriori algorithm:"
                    + str(apriori_recommendations)
                    + "List of products recommended by  Cosine similarity algorithm:"
                    + str(cosine_recommendations)
                    + "List of products recommended by Other customers top selections from the aisle that the customer added the last product are:"
                    + str(location_recommendation)
                    + "List of products recommended of Top items sold in the supermarket (excluding the items added in the cart) are:"
                    + str(top_products_recommendation)
                    + "List of products recommended by Top items bought from same cluster (cluster algorithm) of sellerare:"
                    + str(cluster_recommendations)
                    + "Recommend 5 items (from the list of all the items provided, do not include other items that are not listed) considering complementarity to what is currently in their cart and common sense of what a supermarket cart (shoper behaviour) looks like."
                    + "For the recommendation, prioritize clustering (cluster algorithm), then Apriori Algorithm then Cosine similarity and then the others."
                    + "List the recommendations but not with number 1,2,3,4,5 (don't forget to start with 'Customers with similar selections to yours also added the following products.')."
                    + "Tag in parenthesis which algorithm did you extract the recommendation from.(Top products, Location Recommendation, Apriori Algorithm or Cosine recommendation)"
                    + "Do not include in the recommendation items that could be the same. Example: if 'Fresas' is already in the cart, do not recommend 'Fresas organicas' as an option."
                    + "If strange items appears in recommendations such as 'REAJUSTE SALARIAL' (that don't seem to be sold in a supermarket), do not include in the recommendations "
                    + "After the initial comment provide only the 5 recommended items without any additional explanation or information."
                                        )

              #OpenAI prompt
              client = OpenAI(api_key="xxxx") # mati's API key

              response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                              {"role": "assistant", "content": prompt_recommendation}
                            ]
                          )

              openai_response = str(response.choices[0].message.content)

              print("\n")
              print(response.choices[0].message.content)
              print("\n")

              #translate to english the answer for voice spoken recommendation
              response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english, simplify). It must start with:'Customers with similar selections to yours also added the following products.' Exclude what is included inside the parenthesis (algorithm) " + openai_response}
                            ]
                          )
              tts = gTTS(text=response.choices[0].message.content, lang='en')
              tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_9_more_items.mp3')
              audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_9_more_items.mp3'
              # Reproduce el archivo de audio
              display(Audio(audio_path, autoplay=True))



    print("\nYour final cart:")
    for item in cart:
        print(f"- {item}")



### USER EXPERIENCE ####

def user_experience():

    # Function to get user input
    def get_user_input():
        return input("Enter the product name to add to your cart (or 0 to finish): ").strip()

    ranking_file_path = '/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/General_Ranking/ranking_general.csv'

    top_products = load_ranking_data(ranking_file_path)

    print("\n \n")
    print("Insert the name of the item you included in the cart and press Enter.")
    print("When you are finished, press 0 and Enter to complete your cart.\n")

    cart = []
    top_products_recommendation = display_top_products(top_products, cart)
    for item in top_products_recommendation:
        print(f"{item}")
    print("\n")

    apriori_recommender = AprioriRecommender('/content/drive/Shared drives/Capstone/Dataset_cleaned_merged/Apriori_rules/df_apriori_filtered_rules.csv')

    while True:
        product = get_user_input()
        if product == '0':
            break
        else:
            cart.append(product)
            print("\n")
            print(f"Added {product} to your cart.\n")

            added_product = "Added," + product + "to your cart."

            #OpenAI prompt
            client = OpenAI(api_key="xxx") # mati's API key

            response = client.chat.completions.create(
                          model="gpt-3.5-turbo",
                          messages=[
                            {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english simplify)" + added_product}
                          ]
                        )

            #tts = gTTS(f"Added {product} to your cart", lang='es')
            tts = gTTS(text=response.choices[0].message.content, lang='en')
            tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/cart_items.mp3')
            audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/cart_items.mp3'
            # Reproduce el archivo de audio
            display(Audio(audio_path, autoplay=True))


            print("*" * 50)
            print("Current cart contents:")
            for item in cart:
                print(f"- {item}")
            print("*" * 50)
            print("\n")

            if len(cart) <= 2:
              # Call the recommendation function from recommend_aisle.py
              location_recommendation = display_recommendations(product)
              print("\n Other customers included from the same aisle:")
              for item in location_recommendation:
                      print(f"- {item}")

              print("\n")
              # Display the top products again, excluding items in the cart0
              top_products_recommendation = display_top_products(top_products, cart)
              for item in top_products_recommendation:
                      print(f"{item}")


            # Call the Apriori recommendation function if the cart has at least 4 items
            if len(cart) >= 3 and len(cart) < 9:

              # Call the recommendation function from recommend_aisle.py
              location_recommendation = display_recommendations(product)

              # Display the top products again, excluding items in the cart0
              top_products_recommendation = display_top_products(top_products, cart)

              apriori_recommendations = apriori_recommender.get_recommendations(cart)

              # Call the cosine similarity recommendation function for additional recommendations
              cosine_recommendations = recommender_cosine_similarity(cart,orders=orders, embeddings=embeddings, embeddings_matrix=embeddings_matrix)
              cosine_recommendations = cosine_recommendations[:100]

              prompt_recommendation = (
                    "The recommendation must start by saying, 'Customers with similar selections to yours also added the following products.' "
                    + "Consider that we are building a product recommendation model for a shopper in a supermarket who currently has the following items in their cart: "
                    + str(cart)
                    + ". Our recommendation models would suggest the following: "
                    + "List of products recommended by Apriori algorithm:"
                    + str(apriori_recommendations)
                    + "List of products recommended by  Cosine similarity algorithm:"
                    + str(cosine_recommendations)
                    + "List of products recommended by Other customers top selections from the aisle that the customer added the last product are:"
                    + str(location_recommendation)
                    + "List of products recommended of Top items sold in the supermarket (excluding the items added in the cart) are:"
                    + str(top_products_recommendation)
                    + "Recommend 5 items (from the list of all the items provided, do not include other items that are not listed) considering complementarity to what is currently in their cart and common sense of what a supermarket cart (shoper behaviour) looks like."
                    + "For the recommendation, prioritize Apriori Algorithm and Cosine similarity."
                    + "List the recommendations but not with number 1,2,3,4,5 (don't forget to start with 'Customers with similar selections to yours also added the following products.')."
                    + "Do not include in the recommendation items that could be the same. Example: if 'Fresas' is already in the cart, do not recommend 'Fresas organicas' as an option."
                    + "If strange items appears in recommendations such as 'REAJUSTE SALARIAL' (that don't seem to be sold in a supermarket), do not include in the recommendations "
                    + "After chosing the 5 items and listing them, don't say nothing more."
                                        )

              #OpenAI prompt
              client = OpenAI(api_key="xxx") # mati's API key

              response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                              {"role": "assistant", "content": prompt_recommendation}
                            ]
                          )
              openai_response = str(response.choices[0].message.content)

              print("\n")
              print(response.choices[0].message.content)
              print("\n")

              #translate to english the answer for voice spoken recommendation
              response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english , simplify). It must start with:'Customers with similar selections to yours also added the following products.' " + openai_response}
                            ]
                          )
              tts = gTTS(text=response.choices[0].message.content, lang='en')
              tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_4_more_items.mp3')
              audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_4_more_items.mp3'
              # Reproduce el archivo de audio
              display(Audio(audio_path, autoplay=True))


            # Call the cluster recommendation function if the cart has at least 9 items
            if len(cart) >= 9:

              cluster_recommendations = predict_cluster(cart)

              # Call the cosine similarity recommendation function for additional recommendations
              cosine_recommendations = recommender_cosine_similarity(cart,orders=orders, embeddings=embeddings, embeddings_matrix=embeddings_matrix)
              cosine_recommendations = cosine_recommendations[:100]

              prompt_recommendation = (
                    "The recommendation must start by saying, 'Customers with similar selections to yours also added the following products.' "
                    + "Consider that we are building a product recommendation model for a shopper in a supermarket who currently has the following items in their cart: "
                    + str(cart)
                    + ". Our recommendation models would suggest the following: "
                    + "List of products recommended by Apriori algorithm:"
                    + str(apriori_recommendations)
                    + "List of products recommended by  Cosine similarity algorithm:"
                    + str(cosine_recommendations)
                    + "List of products recommended by Other customers top selections from the aisle that the customer added the last product are:"
                    + str(location_recommendation)
                    + "List of products recommended of Top items sold in the supermarket (excluding the items added in the cart) are:"
                    + str(top_products_recommendation)
                    + "List of products recommended by Top items bought from same cluster (cluster algorithm) of sellerare:"
                    + str(cluster_recommendations)
                    + "Recommend 5 items (from the list of all the items provided, do not include other items that are not listed) considering complementarity to what is currently in their cart and common sense of what a supermarket cart (shoper behaviour) looks like."
                    + "For the recommendation, prioritize clustering (cluster algorithm), then Apriori Algorithm then Cosine similarity and then the others."
                    + "List the recommendations but not with number 1,2,3,4,5 (don't forget to start with 'Customers with similar selections to yours also added the following products.')."
                    + "Do not include in the recommendation items that could be the same. Example: if 'Fresas' is already in the cart, do not recommend 'Fresas organicas' as an option."
                    + "If strange items appears in recommendations such as 'REAJUSTE SALARIAL' (that don't seem to be sold in a supermarket), do not include in the recommendations "
                    + "After the initial comment provide only the 5 recommended items without any additional explanation or information."
                                        )

              #OpenAI prompt
              client = OpenAI(api_key="xxx") # mati's API key

              response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                              {"role": "assistant", "content": prompt_recommendation}
                            ]
                          )

              openai_response = str(response.choices[0].message.content)

              print("\n")
              print(response.choices[0].message.content)
              print("\n")

              #translate to english the answer for voice spoken recommendation
              response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "assistant", "content": "translate to english: (if a product is hard to say, or makes no sense in english, simplify). It must start with:'Customers with similar selections to yours also added the following products.' " + openai_response}
                            ]
                          )
              tts = gTTS(text=response.choices[0].message.content, lang='en')
              tts.save('/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_9_more_items.mp3')
              audio_path = '/content/drive/Shared drives/Capstone/Notebooks/voice_recommendations/recommendations_9_more_items.mp3'
              # Reproduce el archivo de audio
              display(Audio(audio_path, autoplay=True))



    print("\nYour final cart:")
    for item in cart:
        print(f"- {item}")
    print("\n")