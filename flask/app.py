import numpy as np
import pandas as pd
import datetime as dt
import re
import string
import nltk
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Flask utils
from flask import Flask, request, jsonify, render_template, url_for

# Preprocess the data with a function to do all followings: Remove punctuations, tokenize, remove stopwords, and lemmatize
def clean_text(text):
    # Initiate the stopwords and lemmatization
    stopwords = nltk.corpus.stopwords.words('english')
    wn = nltk.WordNetLemmatizer()   
    # Remove punctuation and lowercase the words
    text = "".join([word.lower() for word in text if word not in string.punctuation])    
    # \W matches any non-word character (equivalent to [^a-zA-Z0-9_]). This does not include spaces i.e. \s
    # Add a + just in case there are 2 or more spaces between certain words
    tokens = re.split('\W+', text)    
    # Apply lemmatization and stopwords exclusion within the same step
    text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text

# Function to transform data into count vectorizer form
def count_vect_text(text):
    # Create an instance of CountVectorizer and pass in the clean_text function (defined above) as the analyzer parameter, extracting max_features of 10,000
    count_vect = CountVectorizer(analyzer=clean_text, max_features = 150)
    # Stores the vectorized version of the data in title_text_counts
    count_vect_transform_text = count_vect.fit_transform(text)
    # Expand 'title_text_counts' out to a collection of arrays and then store it in a data frame
    title_text_counts_df = pd.DataFrame(count_vect_transform_text.toarray())
    # Rename the columns with the words
    title_text_counts_df.columns = count_vect.get_feature_names()
    return title_text_counts_df

app = Flask(__name__)

data = pd.read_csv("../datasets/train_modified.csv")

data['label'] = data['label'].map({1:"Fake News", 0:"Real News"})

X = count_vect_text(data['title_text'])
y= data['label']

xgboost_optimal = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                    importance_type='gain', interaction_constraints='',
                    learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                    min_child_weight=1, missing=np.nan, monotone_constraints='()',
                    n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                    tree_method='exact', validate_parameters=1, verbosity=1, eval_metric='mlogloss')

xgboost_optimal.fit(X,y)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        message = request.form['message']
        total_data = count_vect_text([message])
        my_prediction = xgboost_optimal.predict(total_data)
    return render_template('home.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

