## Final Project
### Data Science LHL Project: Restaurant Reccomenation system
### Part 3: Building Model
#By: Chloe Phuong

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Load the dataset
final_restrev_df = pd.read_csv("C:/Users/chloe/OneDrive/Documents/GitHub/LHL-Data-Science-Final-Project/final_restrev_df.csv")

# Lowercasing
final_restrev_df['categories'] = final_restrev_df['categories'].str.lower()
# Removal of Punctuation
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    '''Custom function to remove punctuation'''
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
final_restrev_df['categories'] = final_restrev_df['categories'].apply(lambda text: remove_punctuation(text))
# Removal of Stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    '''Custom function to remove stopwords'''
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])

final_restrev_df['categories'] = final_restrev_df['categories'].apply(lambda text: remove_stopwords(text))

# Lower Casing
final_restrev_df['review_text'] = final_restrev_df['review_text'].str.lower()
## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    '''custom function to remove the punctuation'''
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
final_restrev_df['review_text'] = final_restrev_df['review_text'].apply(lambda text: remove_punctuation(text))
## Removal of Stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    '''custom function to remove the stopwords'''
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])
final_restrev_df['review_text'] = final_restrev_df['review_text'].apply(lambda text: remove_stopwords(text))

# RESTAURANT NAMES:
restaurant_names = list(final_restrev_df['rest_name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]

# Function to get recommendations based on restaurant name and rating tolerance
def restaurant_recommendations(rest_name, rating_tolerance):
    #Setting assertions on the input parameters data types
    assert all(isinstance(rest_name, str) for rest_name in rest_name), 'food_category elements should be strings'
    assert isinstance(rating_tolerance, float), 'rating_tolerance should be a float (i.e. number with a decimal)'

    #Setting an assertion on the rating_tolerance to only accept an input, difference threshold, between 1.0 and 4.0 given the Yelp rating scale of 1 through 5
    assert 1.0 <= rating_tolerance <= 4.0, 'rating_tolerance should be between 1.0 and 4.0, given the Yelp rating scale of 1 through 5'

    # Filter restaurants based on rating tolerance
    filtered_restaurants = final_restrev_df[final_restrev_df['rest_avg_stars'] >= rating_tolerance]

    # Drop duplicates based on rest_name
    filtered_restaurants = filtered_restaurants.drop_duplicates(subset='rest_name', keep='first').reset_index(drop=True)

    # Find the restaurant with the provided rest_name
    target_restaurant = filtered_restaurants[filtered_restaurants['rest_name'] == rest_name]

    if target_restaurant.empty:
        return "The provided restaurant name is not found in the dataset."

    # Calculate the similarity scores based on categories
    category_vectorizer = TfidfVectorizer()
    category_matrix = category_vectorizer.fit_transform(filtered_restaurants['categories'])
    category_similarity = cosine_similarity(category_matrix)

    # Calculate the similarity scores based on review text
    review_vectorizer = TfidfVectorizer()
    review_matrix = review_vectorizer.fit_transform(filtered_restaurants['review_text'])
    review_similarity = cosine_similarity(review_matrix)

    # Combine the similarity scores with a weight of 7:3 (categories:review_text)
    combined_similarity = 0.7 * category_similarity + 0.3 * review_similarity

    # Get the index of the target restaurant within the filtered restaurants dataframe
    rest_index = filtered_restaurants[filtered_restaurants['rest_name'] == rest_name].index[0]

    # Get the similarity scores for all restaurants
    rest_scores = combined_similarity[:, rest_index]

    # Sort the similarity scores
    sorted_indices = rest_scores.argsort()[::-1]

    # Get the top 15 recommended restaurants (excluding the target restaurant itself)
    top_recommendations = filtered_restaurants.iloc[sorted_indices]
    top_recommendations = top_recommendations[top_recommendations.index != rest_index][:15]

    recommended_rest = top_recommendations[['rest_name', 'address', 'rest_avg_stars', 'total_reviews', 'categories', 'review_text']].sort_values(by='rest_avg_stars', ascending=False)

    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR CATEGORIES AND REVIEWS: ' % (str(len(recommended_rest)), rest_name))

    return recommended_rest