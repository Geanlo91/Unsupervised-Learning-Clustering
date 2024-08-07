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
print(data.isnull().sum())

#replace missing values in specific columns
data['How many employees does your company or organization have?, '].fillna(1, inplace=True),
data['Is your employer primarily a tech company/organization?'].fillna(0, inplace=True),
data['Is your primary role within your company related to tech/IT?'].fillna(0, inplace=True),
data['Does your employer provide mental health benefits as part of healthcare coverage?'].fillna('No', inplace=True),
data['Do you know the options for mental health care available under your employer-provided health coverage?'].fillna('No', inplace=True),
data['Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?'].fillna('No', inplace=True),
data['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].fillna('No', inplace=True)

#Print columns handled
print(data.isnull().sum())



