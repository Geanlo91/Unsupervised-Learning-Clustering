import requests
import pandas as pd
import numpy as np
import io
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder


# Load the data
data_file = 'mental-heath-in-tech-2016_20161114.csv'

# Read the data
data = pd.read_csv(data_file)


def descriptive_statistics(data):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # Display the shape of the dataset
    print('Data shape:', data.shape)
    # Display the data types of the columns
    print('Data types:', data.dtypes)
    # Display the number of missing values in each column
    print('Missing values:', data.isnull().sum())
    # Display the summary statistics of the dataset
    print('Summary statistics:', data.describe())
    #rese display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')


    # Plot the distribution of the 'What is your age?' & 'How many employees does your company or organization have?' columns
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    #Age distribution with color differentiation for ages > 80
    age_range = data['What is your age?'].value_counts().sort_index()
    ax[0].bar(age_range.index, age_range.values, color='blue')
    ax[0].bar(age_range[age_range.index > 80].index, age_range[age_range.index > 80].values, color='red')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Age Distribution') 
    # Annotate outliers (ages > 80)
    outliers = age_range[age_range.index > 80]
    for age, count in zip(outliers.index, outliers.values):
        ax[0].text(age, count, 'Outlier', fontsize=12, color='red', ha='center', va='bottom')
        
    #Company size distribution
    sns.countplot(x='How many employees does your company or organization have?', data=data, ax=ax[1])
    ax[1].set_xlabel('Company Size')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('Company Size Distribution')
    plt.tight_layout()
    plt.show()


    #Distribution of the 'What country do you live in?' column and 'What country do you work in?' column in a single barplot for comparison
    df_melted = pd.melt(data, 
                    value_vars=['What country do you live in?', 'What country do you work in?'], 
                    var_name='Country Type', 
                    value_name='Country')

    # Create a bar plot where each country has two bars (one for living, one for working)
    plt.figure(figsize=(15, 6))
    sns.countplot(x='Country', hue='Country Type', data=df_melted)
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.title('Distribution of Country of Residence vs Work')
    plt.tight_layout()
    plt.show()

    return data
data = descriptive_statistics(data)


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

    # Delete rows with missing values in  "What is your age?" column and any age above 80
    data = data.dropna(subset=['What is your age?'])
    data = data.drop(data[data['What is your age?'] > 80].index)
    #check for anomalies in the age column
    #sns.boxplot(data['What is your age?'])
    #plt.title('Age distribution')
    #plt.show()


    # Standardize "What is your gender?" column
    data['What is your gender?'] = data['What is your gender?']\
        .replace(to_replace=r'^[FfWw].*', value='F', regex=True)\
        .replace(to_replace=r'^[Mm].*', value='M', regex=True)

    # Delete rows in the "What is your gender?" column that are not M or F
    data = data.drop(data[(data['What is your gender?'] != 'M') & (data['What is your gender?'] != 'F')].index)

    #standardize the"How many employees does your company or organization have?" column
    data['How many employees does your company or organization have?'] = data['How many employees does your company or organization have?']\
        .replace(to_replace=['26-100', '6-25', '100-500', 'More than 1000', '500-1000', '1-5','unknown'], value=[100, 25, 500, 1500, 1000, 5,1])

    mapping = {
        'Yes': 2, 
        'No': 0, 
        'Maybe': 1, 
        "I don't know": -1, 
        'unknown': -1, 
        'Not eligible for coverage / N/A': 0, 
        'N/A' :0, 
        'I am not sure': 1, 
        'Yes, I think it would' : 2, 
        "No, I don't think it would": 0,
        'None did': 0,
        'Yes, it has': 2,
        'No, it has not': 0,
        'Neutral': 1,
        'No, they do not': 0,
        'Yes, they do': 2,
        'Some did': 1,
        'Yes, they all did': 2,
        'Yes, all of them': 2,
        'None of them': 0,
        'Some of them': 1,
        'Yes, I think they would': 2,
        "No, I don't think they would": 0,
        'No,they do not': 0,
        'Yes, they do': 2,
        'Yes, always': 2,
        'Maybe/Not sure': 1,
        'Yes, I observed': 2,
        'Yes, I experienced': 2,
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
        'N/A (not currently aware)': 0,
        'Some of my previous employers': 1,
        'Yes, at all of my previous employers': 2,
        'No, at none of my previous employers': 0,}


    #convert categorical columns to numerical values from the mapping dictionary
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
            data[col] = data[col].replace(mapping)


    #auto encode the "What country do you live in?" column
    le = LabelEncoder()
    data['What country do you work in?'] = le.fit_transform(data['What country do you work in?'])
    #print the origibal and encoded values
    #print(dict(zip(le.classes_, le.transform(le.classes_))))
          

    #delete the 'what country do you live in?' column
    data = data.drop(['What country do you live in?'], axis=1)


    return data
cleaned_data = preprocess_data(data)



def feature_engineering(cleaned_data):

    mapping1 = { 
        'Yes': 2, 'No': 0, 'Maybe': 1, "I don't know": -1}

    # Create a new column "mental_vs_physical" based on the "Do you think that discussing a mental health disorder with your employer would have negative consequences?" column
    cleaned_data['Negative consequences from discussing mental health with employer'] = cleaned_data['Do you think that discussing a mental health disorder with your employer would have negative consequences?' ].replace(mapping1)
    #delete the original column
    cleaned_data = cleaned_data.drop(['Do you think that discussing a mental health disorder with your employer would have negative consequences?' ], axis=1)

    # Create a new column "Healthwork_benefits" based on the "Have your previous employers provided mental health benefits?" column in int format
    cleaned_data['Have you had mental health benefits before'] = cleaned_data['Have your previous employers provided mental health benefits?']\
        .replace(to_replace=['Some did', 'Yes, they all did', "I don't know",'No, none did','unknown'], value=[1, 2, -1, 0,-1])
    #delete the original column
    cleaned_data = cleaned_data.drop(['Have your previous employers provided mental health benefits?'], axis=1)

    # Create a new column "leave" based on the "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:" column
    cleaned_data['How easy is it to ask for mental health leave?'] = cleaned_data['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:']\
        .replace(to_replace=['Very easy', 'Somewhat easy','Neither easy nor difficult','Somewhat difficult', 'Very difficult','unknown',"I don't know"], value=[5,4,3,2,1,-1,0])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'], axis=1)

    # Create a new column "Mental health effects on work when treated effectively" based on the "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?" column
    cleaned_data['Mental health effects on work when treated effectively'] = cleaned_data['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?']\
        .replace(to_replace=['Not applicable to me', 'Rarely', 'Sometimes', 'Often', 'unknown','Never'], value=[0, 2, 3, 4,-1,1])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?'], axis=1)

    # Create a new column "Mental health effects on work when not treated effectively" based on the "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?" column
    cleaned_data['Mental health effects on work when not treated effectively'] = cleaned_data['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?']\
        .replace(to_replace=['Not applicable to me', 'Rarely', 'Sometimes', 'Often', 'unknown','Never'], value=[0, 2, 3, 4,-1,1])
    #delete the original column
    cleaned_data = cleaned_data.drop(['If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?'], axis=1)

    #Create a new column "Sharing mental health issue with family and friends" based on the "How willing would you be to share with friends and family that you have a mental illness?" column
    cleaned_data['Sharing mental health issue with family and friends'] = cleaned_data['How willing would you be to share with friends and family that you have a mental illness?']\
        .replace(to_replace=['Very open','Somewhat open','Neutral','Somewhat not open','Not applicable to me (I do not have a mental illness)','Not open at all', 'unknown'], value=[5,4,3,2,1,0,-1])
    #delete the original column
    cleaned_data = cleaned_data.drop(['How willing would you be to share with friends and family that you have a mental illness?'], axis=1)


    #one hot encode the "Which of the following best describes your work position?" column. Split the column into multiple columns using | as the separator
    split_positions = cleaned_data['Which of the following best describes your work position?'].str.get_dummies(sep='|')
    # Concatenate the new columns with the original DataFrame
    cleaned_data = pd.concat([cleaned_data, split_positions], axis=1)
    #drop the original and other columns
    cleaned_data.drop(['Which of the following best describes your work position?','Other'], axis=1, inplace=True)


    # Calculate VIF for all features
    vif_data = pd.DataFrame()
    vif_data['feature'] = cleaned_data.columns
    vif_data['VIF'] = [variance_inflation_factor(cleaned_data.values, i) for i in range(len(cleaned_data.columns))]
    
    #drop columns with VIF greater than 10 excluding 'What is your age?' column
    high_VIF = vif_data[vif_data['VIF'] > 10]
    for i in high_VIF['feature']:
        if i != 'What is your age?':
            cleaned_data = cleaned_data.drop([i], axis=1)


    #calculate the correlation matrix
    corr_matrix = cleaned_data.corr()
    #identifying the highly correlated features
    high_threshold = 0.7
    lower_threshold = -0.7
    high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    low_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    high_corr_pairs = high_corr_pairs[(high_corr_pairs > high_threshold) & (high_corr_pairs < 1)]
    low_corr_pairs = low_corr_pairs[(low_corr_pairs < lower_threshold) & (low_corr_pairs > -1)]
    #drop one of the highly correlated features from the pairs, Ensure unique columns are dropped and check for existence
    columns_to_drop = set(i[0] for i in high_corr_pairs.index if i[0] in cleaned_data.columns)
    cleaned_data = cleaned_data.drop(columns=columns_to_drop)
    #drop the lowly correlated columns
    columns_to_drop = set(i[0] for i in low_corr_pairs.index if i[0] in cleaned_data.columns)
    cleaned_data = cleaned_data.drop(columns=columns_to_drop)


    #scale the "How many employees does your company or organization have?" column
    scaler = StandardScaler()
    cleaned_data['How many employees does your company or organization have?'] = scaler.fit_transform(cleaned_data[['How many employees does your company or organization have?']])
    #scale the "What is your age?" column
    cleaned_data['What is your age?'] = scaler.fit_transform(cleaned_data[['What is your age?']])

        
    return cleaned_data

final_data = feature_engineering(cleaned_data)

#Calculate VIF for all features again
vif_data = pd.DataFrame()
vif_data['feature'] = final_data.columns
vif_data['VIF'] = [variance_inflation_factor(final_data.values, i) for i in range(len(final_data.columns))]
print(vif_data)

#Calculate the correlation matrix for features with VIF greater than 3
#high_vif = vif_data[vif_data['VIF'] > 3]
#corr_matrix = final_data[high_vif['feature']].corr()
#sns.heatmap(corr_matrix, annot=True)
#plt.title('Correlation matrix for features with VIF > 3')
#plt.show()



#save the cleaned data to a new csv file
if  'final_data.csv' in os.listdir():
        #overwrite the file
        final_data.to_csv('final_data.csv', index=False)
        print('File already exists, Data cleaning and feature engineering completed successfully')
else:
     final_data.to_csv('final_data.csv', index=False)
















