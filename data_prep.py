import requests
import pandas as pd
import numpy as np
import io

# Load the data
data_file = 'mental-heath-in-tech-2016_20161114.csv'

# Read the data
data = pd.read_csv(data_file)


#check for missing values
missing_values = data.isnull().sum()
#print(data.isnull().sum())

#replace missing values in specific columns
data['How many employees does your company or organization have?'].fillna(1, inplace=True),
data['Is your employer primarily a tech company/organization?'].fillna(0, inplace=True),
data['Is your primary role within your company related to tech/IT?'].fillna(0, inplace=True),
data['Does your employer provide mental health benefits as part of healthcare coverage?'].fillna('No', inplace=True),
data['Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'].fillna('No', inplace=True),
data['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].fillna('No', inplace=True)

#drop specific columns
data = data.drop(['What US state or territory do you live in?', 'What US state or territory do you work in?'], axis=1)

#Print columns handled
#print(data.isnull().sum())

#print columns with missing values
print(data.columns[data.isnull().any()])

#save the cleaned data
#data.to_csv('cleaned_data.csv', index=False)

#Print the cleaned data full table
#print(data)


