# Disaster Response Pipeline Project

## Installation Dependencies
This project uses python 3.7 and the following third party packages:
* flask
* nltk
* numpy
* pandas
* plotly
* sklearn
* sqlalchemy

## Project Motivation
This project explores the rudiments of natural language processing (NLP) machine learning problems.
An anotated dataset of text related to disaster relief efforts was provided to train a machine learning pipeline.
The pipeline cleans and normalizes the data, then tokenizes the text and performs lemmatization on the tokens.
The output is fed into a basic logistic regression algorithm, and paramers are tuned with a grid-search
cross-validation scheme. The model is deployed with a light-weight Flask app that both displays metrics of the
dataset and allows a user to input text for classification.

## File Descriptions
* **app**
  * run.py: Runs the Flask app
  * templates: Templates for running Flask app
* **data**
  * process_data.py: Python script for loading csv data, cleaning, transforming, and then persisting in a sqlite database
  * disaster_messages.csv: Text input for model training
  * disaster_categories.csv: Annotation for supervised machine learning
* **models**
  * train_classifier.py: Creates machine learning pipeline, trains algorithm, and then serializes model for deployment

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/ or [localhost:3001](localhost:3001)

## Acknowledgements
This project was completed in partial fulfillment of the requirements for Udacity's Data Scientist NanoDegree Program. A skeleton of the project was provided by Udacity.

