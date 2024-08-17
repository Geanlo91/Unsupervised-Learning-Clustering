import requests
import pandas as pd
import numpy as np
import io
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



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

    # Delete specific columns with wording that matches the 'Why or why not?' column
    data = data.drop(columns=['Why or why not?', 'Why or why not?.1'])
    

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


    data['How many employees does your company or organization have?'] = data['How many employees does your company or organization have?']\
        .replace(to_replace=['26-100', '6-25', '100-500', 'More than 1000', '500-1000', '1-5','unknown'], value=[100, 25, 500, 1500, 1000, 5,1])

    mapping = {
        '*Yes': 2, 
        'No': 0, 
        'Maybe': 1, 
        "I don't know": 1, 
        'unknown': -1, 
        'Not eligible for coverage / N/A': 0, 
        'N/A' :0, 
        'I am not sure': 1, 
        'Yes, I think it would' : 2, 
        "No, I don't think it would": 0,
        'None did': 0,
        'Some did': 1.5,
        'Yes, they all did': 2,
        'Yes, all of them': 2,
        'None of them': 0,
        'Some of them': 1.5,
        'Yes, I think it would': 2,
        "No, I don't think it would": 0,
        'No,they do not': 0,
        'Yes, they do': 2,
        'Maybe/Not sure': 1,
        'Yes, I observed': 2,
        #map empty strings to unknown
        '': 1,
        'M': 1,
        'F': 0,
        'Always': 2,
        'Sometimes': 1,
        'Never': 0,
        'Yes, I was aware of all of them': 2,
        'I was aware of some': 1,
        'No, I only became aware later': 0,
        'N/A (not currently aware)': 0,}

    #convert categorical columns to numerical values from the mapping dictionary
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
            data[col] = data[col].replace(mapping)

    return data
cleaned_data = preprocess_data(data)




def feature_engineering(cleaned_data):

    mapping1 = { 
        'Yes': 2, 'No': 0, 'Maybe': 1, "I don't know": 1}

    # Create a new column "mental_vs_physical" based on the "Do you think that discussing a mental health disorder with your employer would have negative consequences?" column
    cleaned_data['mental_health_work_fears'] = cleaned_data['Do you think that discussing a mental health disorder with your employer would have negative consequences?' ].replace(mapping1)
    #delete the original column
    cleaned_data = cleaned_data.drop(['Do you think that discussing a mental health disorder with your employer would have negative consequences?' ], axis=1)

    # Create a new column "Healthwork_benefits" based on the "Have your previous employers provided mental health benefits?" column in int format
    cleaned_data['Healthwork_benefits'] = cleaned_data['Have your previous employers provided mental health benefits?']\
        .replace(to_replace=['Some did', 'Yes, they all did', "I don't know",'No, none did','unknown'], value=[1.5, 2, 1, 0,-1])
    #delete the original column
    cleaned_data = cleaned_data.drop(['Have your previous employers provided mental health benefits?'], axis=1)

    # Create a new column "leave" based on the "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:" column
    cleaned_data['Getting_Mental_health_leave'] = cleaned_data['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:']\
        .replace(to_replace=['Very easy', 'Somewhat easy','Neither easy nor difficult','Somewhat difficult', 'Very difficult','unknown',"I don't know"], value=[2,1,0,-1,-2,-3,-3])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'], axis=1)

    # Create a new column "Mental health effects on work when treated effectively" based on the "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?" column
    cleaned_data['Mental health effects on work when treated effectively'] = cleaned_data['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?']\
        .replace(to_replace=['Not applicable to me', 'Rarely', 'Sometimes', 'Often', 'unknown','Never'], value=[-1, 2, 3, 4,-1,0])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'], axis=1)

    # Create a new column "Mental health effects on work when not treated effectively" based on the "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?" column
    cleaned_data['Mental health effects on work when not treated effectively'] = cleaned_data['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?']\
        .replace(to_replace=['Not applicable to me', 'Rarely', 'Sometimes', 'Often', 'unknown','Never'], value=[-1, 1, 2, 3,-1,0])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'], axis=1)

#one hotencode the "Which of the following best describes your work position?" column. Split the column into multiple columns using | as the separator

    split_positions = cleaned_data['Which of the following best describes your work position?'].str.get_dummies(sep='|')

    # Concatenate the new columns with the original DataFrame
    cleaned_data = pd.concat([cleaned_data, split_positions], axis=1)

    #drop the original column
    cleaned_data.drop('Which of the following best describes your work position?', axis=1, inplace=True)
    #drop the "other" column
    cleaned_data.drop('Other', axis=1, inplace=True)

        
    
    return cleaned_data

final_data = feature_engineering(cleaned_data)


#save the cleaned data to a new csv file
final_data.to_csv('final_data.csv', index=False)
print('Data cleaning and preprocessing complete!')

#print unique values in the " What country do you live in?" column
print(final_data['What country do you live in?'].unique())














