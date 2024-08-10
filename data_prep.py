import requests
import pandas as pd
import numpy as np
import io
import seaborn as sns


# Load the data
data_file = 'mental-heath-in-tech-2016_20161114.csv'

# Read the data
data = pd.read_csv(data_file)


#Understanding the data and missing data volumes
missing_values = data.isnull().sum().sort_values(ascending=False)
missing_percentage = (data.isnull().sum() / len(data)) * 100
pd.set_option('display.max_rows', None)
#print(pd.concat([missing_values, missing_percentage,data.dtypes], axis=1, keys=['Missing Values', 'Percentage','Data Type']))


#identify columns in float format and convert to int
float_cols = data.select_dtypes(include=['float64']).columns
for col in float_cols:
    data[col] = data[col].fillna(0.0).astype(np.int64)

#Drop columns with more than 60% missing values
data = data.dropna(thresh=data.shape[0]*0.6, axis=1)

#Delete specific columns
data = data.drop(['Why or why not?'], axis=1)

#Replace missing values in int columns with zero
int_cols = data.select_dtypes(include=['int64']).columns
for col in int_cols:
    if col != 'What is your age?':
        data[col] = data[col].fillna(0).astype(np.int64)


#Delete rows with missing "What is your age?" values
data = data.dropna(subset=['What is your age?'])

#Standardize "What is your gender?"" column. Replace any word that starts with F of f with F and any that starts with M or m with M
data['What is your gender?'] = data['What is your gender?']\
    .replace(to_replace=r'^[FfWw].*', value='F', regex=True)\
    .replace(to_replace=r'^[Mm].*', value='M', regex=True)

#Delete rows in the "What is your gender?" column that are not M or F
data = data.drop(data[(data['What is your gender?'] != 'M') & (data['What is your gender?'] != 'F')].index)
print(data['What is your gender?'])



#Feature generation
#Create a new column that will have percentage float value of missing values in each row
data['missing_values'] = data.isnull().sum(axis=1) / len(data.columns) * 100
#Print missing values column for rows with more than 40% missing values

#Drop rows with more than 40% missing values
data = data.drop(data[data['missing_values'] > 40].index)














