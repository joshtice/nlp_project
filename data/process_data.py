#!/usr/bin/env python3

"""
File:     process_data.py
Author:   Joshua Tice
Date:     May 15, 2019

This script imports data from two CSV files, cleans the data, and then
uploads the data to a sqlite database in preparation for training a
natural language processing machine learning algorithm.

Requirements
------------
pandas
sqlalchemy

Acknowledgements
----------------
The main() function was provided by Udacity
"""

import sys

from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Imports text data from CSVs into a pandas dataframe

    Parameters
    ----------
    messages_filepath : str
        Path to CSV file containing message text
    categories_filepath : str
        Path to CSV file containing classification information

    Returns
    -------
    pandas.DataFrame
        Dataframe with concatenated message and classification data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df



def clean_data(df):
    """
    Clean data for machine learning task

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be cleaned

    Returns
    -------
    pandas.DataFrame
        The cleaned dataframe
    """

    # categories originally exist as a single text field containing the label
    # and binary value

    # remove labels and make into column names
    categories = df.categories.str.split(";", expand=True)
    col_names = categories.iloc[0].apply(lambda x: x[:-2])
    categories.columns = col_names

    # extract the binary values from the text field
    no_info_cols = []
    for col in categories:
        categories[col] = categories[col].str[-1]
        categories[col] = categories[col].astype(int)
        categories[col] = categories[col].apply(lambda x: 0 if x==0 else 1)
        if categories[col].max() == 0:
            no_info_cols.append(col)

    if no_info_cols:
        categories = categories.drop(labels=no_info_cols, axis=1)

    # remove the original columns
    df = df.drop(labels=['id', 'original', 'categories'], axis=1)
    df = pd.concat([df, categories], axis=1, sort=False)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Persist dataframe in database

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be saved to disk
    database_filename : str
        Path to the database where the data will be persisted
    """

    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('message_data', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()