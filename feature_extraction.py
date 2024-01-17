import pandas as pd
from datetime import datetime
import re
import string

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import numpy as np
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from math import nan
#############################################################################
# Helper function
# Convert the timestamps to day-month-year and real time 
def timestamp_to_date(df):
    print("Processing timestamp")
    time_arr = []
    day_arr = []
    year_arr = []
    month_arr = []

    for val in df["Time"]:
        date_time = str(datetime.fromtimestamp(val))
        date = date_time.split(" ")[0]
        time = date_time.split(" ")[1]

        year = date.split("-")[0]
        month = date.split("-")[1]
        day = date.split("-")[2]

        time_arr.append(time)

        day_arr.append(day)
        month_arr.append(month)
        year_arr.append(year)

    # df["Real_Time"] = np.array(time_arr)

    df["Year"] = np.array(year_arr).astype(int)
    df["Month"] = np.array(month_arr).astype(int)
    df["Day"] = np.array(day_arr).astype(int)

    return df  

# Deal with text stemmer
# col: either "Text", or "Summary"
def text_stem(df, col):
    snowball_stemmer = SnowballStemmer(language='english')
    
    # Stemmed words
    stemmed_words = []    
    for _, value in df[col].iteritems():
        tokenized_article = word_tokenize(value)

        stemmed_article = ''
        for j in range(len(tokenized_article)):
            word = snowball_stemmer.stem(tokenized_article[j])
            stemmed_article += " " + word

        stemmed_words.append(stemmed_article)
        
    df[f'{col}_Stemmed'] = np.array(stemmed_words)
    return df

# Find the average score of 
# The Product
# The User usually given
def assign_average(df, col):
    print(f"Processing score and {col}")
    score_average = df[["Score", col]]
    mean_score = score_average.groupby(col)["Score"].mean()

    mean_arr = []
    for product in score_average[col]:
        mean_arr.append(mean_score.loc[product])

    return np.array(mean_arr)

# Length of the Review
def review_length(df, col):
    print(f"Processing review length and {col}")
    review_length = []
    for row in df[col]:
        review_length.append(len(row.split()))
    
    return np.array(review_length)

# Use Tf-IDF to product the weight of each word
def text_process(df):
    # Process the text
    alphanumeric = lambda x: re.sub(r"""\w*\d\w*""", ' ', x)
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

    print("Process text and summary column")
    # Take the Text and Summary columns from the data
    df["Text"] = df["Text"].fillna("").map(alphanumeric).map(punc_lower)
    df["Summary"] = df["Summary"].fillna("").map(alphanumeric).map(punc_lower)
    print("Done process text and summary column")

    # Stemmed the words from the Text and Summary Column
    # df = text_stem(df, "Summary")
    
    # print(f"Processing stemmed")
    # snowball_stemmer = SnowballStemmer(language='english')
    # # Stemmed words
    # stemmed_words = []    
    # for _, value in df["Summary"].iteritems():
    #     tokenized_article = word_tokenize(value)

    #     stemmed_article = ''
    #     for j in range(len(tokenized_article)):
    #         word = snowball_stemmer.stem(tokenized_article[j])
    #         stemmed_article += " " + word

    #     df["Summary"].replace(to_replace=value, value=stemmed_article, inplace=True)
    #     stemmed_words.append(stemmed_article)
    # print(f"Finish stemming")

    print(f"Executing the TfIDF vectorizer on Text")
    # Tfidf vectorizer for the text column
    vectorizer_text = TfidfVectorizer(min_df=0.1, max_df=0.85)
    text_full = vectorizer_text.fit_transform(df["Text"])

    # text data taken from the vectorizer after transforming the text
    text_data = pd.DataFrame(text_full.todense(), 
                        columns=vectorizer_text.get_feature_names_out(), 
                        index=df.index)

    print("Concating Vectorizer and current df")
    df = pd.concat([df, text_data], axis=1)
    print("Finishing Concat")

    return df

def trustworthy(df): 
    trust = df[["Score", "Helpfulness"]]
    trust_arr = []
    print("Process Trustworthy Comments")
    for i in range(len(trust)):
        score_i = trust["Score"].iloc[i]
        helpful_i = trust["Helpfulness"].iloc[i]

        if (np.isnan(score_i)) or (np.isnan(helpful_i)):
            trust_arr.append(0)

        elif (score_i >= 0.0 and score_i <= 3.5):
            if (helpful_i < 0.65):
                trust_arr.append(0)
            else: 
                trust_arr.append(1)

        elif (score_i >= 3.5 and score_i <= 5.0):
            if (helpful_i >= 0.65):
                trust_arr.append(1)
            else: 
                trust_arr.append(0)
        
    return np.array(trust_arr)
#############################################################################

def process(df):
    # This is where you can do all your processing
    print("Computing helpfulness")
    # Helpfulness 
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    print("Done computing helpfulness")
    print()

    # Statistics Analysis
    # Timestamp
    print("Computing Timestamp")
    df = timestamp_to_date(df)
    df["Year"] = df["Year"].fillna(0)
    df["Month"] = df["Month"].fillna(0)
    df["Day"] = df["Day"].fillna(0)
    print("Done computing timestamp")
    print()

    # Average review for a product
    print("Computing product average review")
    df["Product_Average_Review"] = assign_average(df, "ProductId")
    df["Product_Average_Review"] = df["Product_Average_Review"].fillna(0)
    print("Done computing average product")
    print()

    # Average review rate of a given user
    print("Computing user average review")
    df["User_Average_Review"] = assign_average(df, "UserId")
    df["User_Average_Review"] = df["User_Average_Review"].fillna(0)
    print("Done computing average user")
    print()

    # Text
    # Length of review indicates the quality of the product
    print("Computing review length")
    df["Text"] = df["Text"].fillna("")
    df["Review_Length"] = review_length(df, "Text")
    print("Done computing review length")
    print() 

    print("Computing Trustworthy review")
    df["Trustworthy"] = trustworthy(df)
    print("Done Computing Trusthworthy reivew")

    print("Computing Text extract")
    # Using the helper function to process the text
    df = text_process(df)
    print("Done Computing Text extract")
    print("Returning the dataframe")

    return df


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)
