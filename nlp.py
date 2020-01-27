# *************************************************
# HEADER
# *************************************************

# -*- coding: utf-8 -*-
"""

Filename: name_iteration_keyword.ipynb

Author:   Ednalyn C. De Dios
Phone:    (210) 236-2685
Email:    ednalyn.dedios@taskus.com 

Created:  January 00, 2020
Updated:  January 00, 2020

PURPOSE: describe the purpose of this script.

PREREQUISITES: list any prerequisites or
assumptions here.

DON'T FORGET TO:
1. Action item.
2. Another action item.
3. Last action item.

"""



# *************************************************
# ENVIRONMENT
# *************************************************

# for reading files from the local machine
import os

# for manipulating dataframes
import pandas as pd
import numpy as np

# natural language processing
import re
import unicodedata
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = ['campaign']

# visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# to print out all the outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# to print out all the columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# *************************************************
# BASIC CLEAN
# *************************************************

def clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]



# *************************************************
# DISPLAY MISSING VALUES
# *************************************************

def missing_values_col(df):
    """
    Write or use a previously written function to return the
    total missing values and the percent missing values by column.
    """
    null_count = df.isnull().sum()
    null_percentage = (null_count / df.shape[0]) * 100
    empty_count = pd.Series(((df == ' ') | (df == '')).sum())
    empty_percentage = (empty_count / df.shape[0]) * 100
    nan_count = pd.Series(((df == 'nan') | (df == 'NaN')).sum())
    nan_percentage = (nan_count / df.shape[0]) * 100
    return pd.DataFrame({'num_missing': null_count, 'missing_percentage': null_percentage,
                         'num_empty': empty_count, 'empty_percentage': empty_percentage,
                         'nan_count': nan_count, 'nan_percentage': nan_percentage})



# *************************************************
# READ CSV FILE
# *************************************************

df = pd.read_csv('../data/campaign_nlp.csv')



# *************************************************
# READ EXCEL FILE
# *************************************************

df = pd.read_excel('../data/campaign_nlp.xlsx')



# *************************************************
# READ ALL FILES WITHIN A FOLDER
# *************************************************

def read_data(input_folder):
    '''
    This function reads each the raw data files as dataframes and
    combines them into a single data frame.
    '''
    for i, file_name in enumerate(os.listdir(input_folder)):
        try:
            # df = pd.read_excel(os.path.join(input_folder, file_name))
            df = pd.read_csv(os.path.join(input_folder, file_name))
            df['file_name'] = file_name
            if i == 0:
                final_df = df.copy()
            else:
                final_df = final_df.append(df)

        except Exception as e:
            print(f"Cannot read file: {file_name}")
            print(str(e))
    return final_df

input_folder = 'G:/path/to/data/parent_folder_name'
input_df = read_data(input_folder)