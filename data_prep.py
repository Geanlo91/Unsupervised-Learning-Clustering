import requests
import pandas as pd
import numpy as np
import io
import seaborn as sns


# Load the data
data_file = 'mental-heath-in-tech-2016_20161114.csv'

# Read the data
data = pd.read_csv(data_file)


def preprocess_data(data, missing_threshold=0.6):
    # Understanding the data and missing data volumes
    missing_values = data.isnull().sum().sort_values(ascending=False)
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    pd.set_option('display.max_rows', None)
    
    # Identify columns in float format and convert to int
    float_cols = data.select_dtypes(include=['float64']).columns
    for col in float_cols:
        data[col] = data[col].fillna(0.0).astype(np.int64)

    # Drop columns with more than 60% missing values
    data = data.dropna(thresh=len(data) * missing_threshold, axis=1)

    # Delete specific columns
    if 'Why or why not?' in data.columns:
        data = data.drop(['Why or why not?'], axis=1)

    # Replace missing values in int columns with zero and place unknown in object columns
    int_cols = data.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if col != 'What is your age?':
            data[col] = data[col].fillna(0).astype(np.int64)

    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        data[col] = data[col].fillna('unknown')

    # Delete rows with missing "What is your age?" values
    data = data.dropna(subset=['What is your age?'])

    # Standardize "What is your gender?" column
    data['What is your gender?'] = data['What is your gender?']\
        .replace(to_replace=r'^[FfWw].*', value='F', regex=True)\
        .replace(to_replace=r'^[Mm].*', value='M', regex=True)

    # Delete rows in the "What is your gender?" column that are not M or F
    data = data.drop(data[(data['What is your gender?'] != 'M') & (data['What is your gender?'] != 'F')].index)

    
    return data

cleaned_data = preprocess_data(data)


def feature_engineering(cleaned_data):

    mapping = { 
        'Yes': 2, 'No': 0, 'Maybe': 1, "I dont't know": -1, 'unknown': -2}

    # Create a new column "mental_vs_physical" based on the "Do you think that discussing a mental health disorder with your employer would have negative consequences?" column
    cleaned_data['mental_health_work_fears'] = cleaned_data['Do you think that discussing a mental health disorder with your employer would have negative consequences?' ].map(mapping)

    # Create a new column "Healthwork_benefits" based on the "Have your previous employers provided mental health benefits?" column in int format
    cleaned_data['Healthwork_benefits'] = cleaned_data['Have your previous employers provided mental health benefits?']\
        .replace(to_replace=['Some did', 'Yes, they all did', "I don't know",'No, none did'], value=[2, 1, -1, 0])

    # Create a new column "leave" based on the "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:" column
    cleaned_data['Mentalhealth_leave'] = cleaned_data['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:']\
        .replace(to_replace=['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'], value=[5, 4, 3, 2, 1])

    return cleaned_data

final_data = feature_engineering(cleaned_data)

#show number of missing values for each column
print(final_data.isnull().sum())













