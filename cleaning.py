import pandas as pd
import numpy as np
from sqlalchemy import create_engine


amazon_reviews = pd.read_csv("../archive/Train.csv")
print(amazon_reviews.head())


engine = create_engine(' database connection URL ')

amazon_reviews.to_sql(name='amazon_reviews',con=engine,if_exists='fail',index=False)


amazon_reviews = amazon_reviews.drop_duplicates()

amazon_reviews.isnull().sum()
# enought data, drop Nan in reviewText summary 
amazon_reviews.dropna(subset=["reviewText", "summary", "label"], inplace=True)
amazon_reviews.isnull().sum()

amazon_reviews.info()

amazon_reviews = amazon_reviews.drop(["related","songs"], axis=1)

# Helpful column manipulation 
import ast
amazon_reviews['helpful_list'] = amazon_reviews['helpful'].apply(ast.literal_eval)
amazon_reviews['helpful_votes'] = amazon_reviews['helpful_list'].apply(lambda x: x[0])
amazon_reviews['all_votes'] = amazon_reviews['helpful_list'].apply(lambda x : x[1])

amazon_reviews[["all_votes", "helpful_votes"]]
amazon_reviews["helpful_votes_proc"] = amazon_reviews["helpful_votes"] / amazon_reviews["all_votes"]

amazon_reviews["helpful_votes_proc"] = amazon_reviews["helpful_votes_proc"].fillna(0)
amazon_reviews["helpful_votes_proc"].isna().sum()

amazon_reviews = amazon_reviews.drop(["helpful_list","helpful"], axis=1)
amazon_reviews.columns

# drop any not related to overall columns of type string 
amazon_reviews = amazon_reviews.drop(["label", "categories", "root-genre", "reviewTime"], axis=1)
amazon_reviews.info()


min_length = amazon_reviews["reviewText"].apply(len).min()
print(min_length)

# left only review of length >= 10 
amazon_reviews = amazon_reviews[amazon_reviews["reviewText"].apply(len) >= 10]
amazon_reviews.shape
print(amazon_reviews)

amazon_reviews.info()


amazon_reviews.to_sql(name='amazon_reviews_cleaned', con=engine, if_exists='replace', index=False)

########## LANGUAGE DETECTION 
from langdetect import detect_langs
languages = [] 

for row in range(len(amazon_reviews)):
    languages.append(detect_langs(amazon_reviews.iloc[row, 3])) 

# Clean the list by splitting     
languages = [str(lang).split(':')[0][1:] for lang in languages]

from collections import Counter
print(Counter(languages))
amazon_reviews.shape

amazon_reviews["language_review"] = languages

amazon_reviews = amazon_reviews[amazon_reviews['language_review'] == 'en']
amazon_reviews.shape
amazon_reviews = amazon_reviews.drop("language_review", axis=1)
amazon_reviews.head()


#################### Cleaning text: lowercase, punctuation etc
amazon_reviews['combined_text'] = amazon_reviews['reviewText'] + " " + amazon_reviews['summary']


import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
stemmer = PorterStemmer()
def clean_text(text):
    # Remove punctuation
    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text_lower = text_no_punctuation.lower()
    tokens = word_tokenize(text_lower)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = ' '.join(stemmed_tokens)
    return stemmed_text

amazon_reviews['combined_text'] = amazon_reviews['combined_text'].apply(clean_text)

amazon_reviews['combined_text'].head()
amazon_reviews.columns
amazon_reviews.info()

amazon_reviews.drop(columns=['reviewText', 'summary'], inplace=True)

############ new column 'sentiment' creation ################

amazon_reviews["overall"].value_counts()

def sentiment_based_on_overall(row):
    # Classify sentiment based on 'overall' rating
    if row['overall'] > 3:
        return 1  # Positive sentiment
    else:
        return 0  # Negative sentiment

amazon_reviews['sentiment'] = amazon_reviews.apply(sentiment_based_on_overall, axis=1)

amazon_reviews.head()

amazon_reviews.to_sql(name='amazon_reviews_cleaned_eng', con=engine, if_exists='replace', index=False)
