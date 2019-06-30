#!/usr/bin/env python3

"""
File:     train_classifier.py
Author:   Joshua Tice
Date:     May 15, 2019

This script imports data from a sqlite database and then trains a
a machine learning model to classify text related to disaster relief
efforts. After training and evaluating the model, the script then saves
the model to a pickle file for deployment.

Requirements
------------
nltk
numpy
pandas
sklearn
sqlalchemy

Acknowledgements
----------------
The main() function was kindly provided by Udacity
"""

import pickle
import re
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download([
    'averaged_perceptron_tagger',
    'punkt',
    'stopwords',
    'wordnet', ])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """
    Load data from a specified database into numpy arrays

    Parameters
    ----------
    database_filepath : str
        Path to the database where data is extracted from

    Returns
    -------
    tuple of numpy arrays

    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='message_data', con=engine)
    X = df['message'].values
    Y = df.iloc[:, 2:].values
    category_names = df.columns[2:]

    return X, Y, category_names


def tokenize(text):
    """
    Process input text by (i) normalizing, (ii) tokenizing, and
    (iii) lemmatizing word tokens

    Parameters
    ----------
    text : str
        Raw text input to process

    Returns
    -------
    list
        tokenized elements of text
    """

    # Normalize text
    substitutions = [
        (re.compile(r"(?:http[s]?://)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"), 'url'),
        (re.compile(r"(?:www.)?[.\w\d]+(?:.com|.org|.net|.gov)"), 'link'),
        (re.compile(r"http [\w\d.]+ [.\w\d]+"), 'link'),
        (re.compile(r"[^-\s\w\d]+"), ' '),
    ]

    for regex, substitution in substitutions:
        text = re.sub(regex, substitution, text)
    text = text.lower().strip()

    # Tokenize text and remove stopwords
    tokens = [
        word for word in word_tokenize(text)
        if word not in stopwords.words('english')
    ]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token, 'v') for token in tokens]

    return tokens


def build_model():
    """
    Instantiate a machine learning pipeline
    """

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            LogisticRegression(multi_class='ovr', solver='lbfgs')
        )),
    ])

    # use F10 score when training the model - more important to grab all the 
    # emergency messages and occasionally misclassify a non-urgent message
    # rather than make sure the urgent messages are absolutely correctly
    # classified
    f10_micro = make_scorer(fbeta_score, beta=10, average='micro')

    params = {
        'clf__estimator__C': [1.0, 10.0, 100.0],
    }

    model = GridSearchCV(estimator=pipeline, param_grid=params,
        scoring=f10_micro, cv=5, n_jobs=-1)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print metrics after evaluating model

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A trained machine learning model
    X_test : numpy.array
        Test input for the model
    Y_test : numpy.array
        True test output
    category_names : list
        List of possible categories
    """

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.flatten(), Y_pred.flatten()))

    # print classification report for each category
    for i, category in enumerate(category_names):
        print(category.center(52, '-'), '\n')
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Persist the machine learning model to a serialized pickle file

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A trained machine learning model
    model_filepath : str
        Path to the pickle file to save the model
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()