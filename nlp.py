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



# *************************************************
# DISPLAY THE FIRST AND LAST 5 ROWS OF A DATAFRAME
# *************************************************

df.head()
df.tail()



# *************************************************
# DISPLAY THE FIRST 10 ITEMS OF A LIST
# *************************************************

my_list[:10]




# *************************************************
# USE ILOC TO SELECT ROWS
# *************************************************

# Single selections using iloc and DataFrame

# Rows:
data.iloc[0] # first row of data frame (Aleshia Tomkiewicz) - Note a Series data type output.
data.iloc[1] # second row of data frame (Evan Zigomalas)
data.iloc[-1] # last row of data frame (Mi Richan)

# Columns:
data.iloc[:,0] # first column of data frame (first_name)
data.iloc[:,1] # second column of data frame (last_name)
data.iloc[:,-1] # last column of data frame (id)

# Multiple row and column selections using iloc and DataFrame

data.iloc[0:5] # first five rows of dataframe
data.iloc[:, 0:2] # first two columns of data frame with all rows
data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
data.iloc[0:5, 5:8] # first 5 rows and 5th, 6th, 7th columns of data frame (county -> phone1).



# *************************************************
# USE LOC TO SELECT ROWS
# *************************************************

# Select rows with first name Ednalyn, include all columns between 'city' and 'email'
data.loc[data['first_name'] == 'Ednalyn', 'city':'email']
 
# Select rows where the email column ends with 'gmail.com', include all columns
data.loc[data['email'].str.endswith("gmail.com")]   
 
# Select rows with first_name equal to some values, all columns
data.loc[data['first_name'].isin(['Ednalyn', 'Ederlyne', 'Edelyn'])]   
       
# Select rows with first name Ednalyn and gmail email addresses
data.loc[data['email'].str.endswith("gmail.com") & (data['first_name'] == 'Ednalyn')] 
 
# select rows with id column between 100 and 200, and just return 'zip' and 'web' columns



# *************************************************
# SELECTING COLUMNS
# *************************************************

dfx = df[['column_name',
          '',
          '',
          '',
          ''
        ]]




# *************************************************
# DROPPING NULL VALUES
# *************************************************

df = df.dropna()


# *************************************************
# NAMING COLUMNS
# *************************************************

df.columns=['column1','column2','column3']



# *************************************************
# RENAMING COLUMNS
# *************************************************

df = df.rename(columns={'old_name':'new_name',
                        '':'',
                        '':'',
                        '':'',
                        '':''
                        })



# *************************************************
# DROPPING DUPLICATES
# *************************************************

# drops duplicate values in column id
dfx = df.drop_duplicates(subset ="column_id", keep = False) 



# *************************************************
# SELECTING NON-NULL VALUES
# *************************************************

dfx = df.loc[df['column_name'].notnull()]



# *************************************************
# VALUE COUNTS
# *************************************************

labels = pd.concat([df.rating.value_counts(),
                    df.rating.value_counts(normalize=True)], axis=1)
labels.columns = ['n', 'percent']
labels



# *************************************************
# SHAPE AND LENGTH
# *************************************************

df.shape
len(some_list)



# *************************************************
# INFO AND DESCRIBE
# *************************************************

df.info()
df.describe



# *************************************************
# MERGING DATAFRAMES
# *************************************************

df_merged = df1.merge(df2,
                      left_on='id1',
                      right_on='id2',
                      suffixes=('_left', '_right'))



# *************************************************
# WORKING WITH TIMESTAMPS
# *************************************************

df.timestamp[:1]

dtz = []
for ts in df.timestamp:
  dtz.append(parse(ts))
dtz[:10]

df['date_time_zone'] = df.apply(lambda row: parse(row.timestamp), axis=1)

df.set_index('date_time_zone', inplace=True)



# *************************************************
# DESIGNATING CSAT AND DSAT
# *************************************************

# creates a new column and designates a row as either high or low
df['csat'] = np.where(df['rating']>=3, 'high', 'low')



# *************************************************
# SPLITTING CSAT AND DSAT
# *************************************************

df_positive =  df.loc[df['column_name'] == 'positive']
df_negative =  df.loc[df['column_name'] == 'negative']



# *************************************************
# TRANSFORMING DF COLUMN INTO A LIST OF CLEAN WORDS
# *************************************************

my_list = df.column.tolist()
my_words = clean(''.join(str(good_list)))



# *************************************************
# N-GRAMS RANKING
# *************************************************

def get_words(df,column):
    """
    Takes in a dataframe and columns and returns a list of
    words from the values in the specified column.
    """
    return clean(''.join(str(df[column].tolist())))

def get_unigrams(words):
    """
    Takes in a list of words and returns a series of
    unigrams with value counts.
    """
    return  pd.Series(words).value_counts()

def get_bigrams(words):
    """
    Takes in a list of words and returns a series of
    bigrams with value counts.
    """
    return (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]

def get_trigrams(words):
    """
    Takes in a list of words and returns a series of
    trigrams with value counts.
    """
    return (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]

def get_qualgrams(words):
    """
    Takes in a list of words and returns a series of
    qualgrams with value counts.
    """
    return (pd.Series(nltk.ngrams(words, 4)).value_counts())[:20]

def get_ngrams(df,column):
    """
    Takes in a dataframe with column name and generates a
    dataframe of unigrams, bigrams, trigrams, and qualgrams.
    """
    return get_bigrams(get_words(df,column)).to_frame().reset_index().rename(columns={'index':'bigram','0':'count'}), \
           get_trigrams(get_words(df,column)).to_frame().reset_index().rename(columns={'index':'trigram','0':'count'}), \
           get_qualgrams(get_words(df,column)).to_frame().reset_index().rename(columns={'index':'qualgram','0':'count'})



# *************************************************
# N-GRAMS VIZ
# *************************************************

def viz_bigrams(df,column):
    get_bigrams(get_words(df,column)).sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

    plt.title('20 Most Frequently Occuring Bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurances')

    ticks, _ = plt.yticks()
    labels = get_bigrams(get_words(df,column)).reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1]).iloc[::-1]
    _ = plt.yticks(ticks, labels)

def viz_trigrams(df,column):
    get_trigrams(get_words(df,column)).sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

    plt.title('20 Most Frequently Occuring Trigrams')
    plt.ylabel('Trigram')
    plt.xlabel('# Occurances')

    ticks, _ = plt.yticks()
    labels = get_trigrams(get_words(df,column)).reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1] + ' ' + t[2]).iloc[::-1]
    _ = plt.yticks(ticks, labels)
    
def viz_qualgrams(df,column):
    get_bigrams(get_words(df,column)).sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))

    plt.title('20 Most Frequently Occuring Qualgrams')
    plt.ylabel('Qualgram')
    plt.xlabel('# Occurances')

    ticks, _ = plt.yticks()
    labels = get_qualgrams(get_words(df,column)).reset_index()['index'].apply(lambda t: t[0] + ' ' + t[1] + ' ' + t[2] + ' ' + t[3] ).iloc[::-1]
    _ = plt.yticks(ticks, labels)



# *************************************************
# MANUAL CRITERIA SEARCH
# *************************************************

# Create an empty list 
overall_criteria_list =[] 

for index, row in df.iterrows():
    if ('term1' in row['column_name'] and 'term2' in row['column_name']):
        overall_criteria_list .append([row.column1,
                                        row.column2,
                                        row.column3,
                                        row.column4,
                                        row.column5
                                        ])