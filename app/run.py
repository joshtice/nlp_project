"""
File:     run.py
Date:     May 18, 2019

This script runs a simple Flask app that displays information from the
Disaster Response data set and allows the user to enter arbitrary text
for classification by a machine learning algorithm trained on the data.

Requirements
------------
flask
nltk
pandas
plotly
sklearn
sqlalchemy

Acknowledgements
----------------
The main() function was provided by Udacity
"""

import json

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download([
    'averaged_perceptron_tagger',
    'punkt',
    'stopwords',
    'wordnet', ])
from flask import Flask
from flask import render_template, request, jsonify
import pandas as pd
import plotly
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """
    Tokenize string for machine learning prediction

    Parameters
    ----------
    text : str
        Input text to be tokenized

    Returns
    -------
    list
        Tokens extracted from input text
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    class_summary = df.loc[:, 'related':'direct_report'].sum().sort_values(ascending=False)
    class_counts = class_summary.values
    class_names = class_summary.index

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classifications',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Classification'
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()