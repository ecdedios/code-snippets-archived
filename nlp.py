# HEADER

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

# ENVIRONMENT

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