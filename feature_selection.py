from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv('final_data.csv')
print(data.head())

#Prepare the data
X = data

#Creating a pseudo-target for Random Forest (using random labels)
np.random.seed(42)
pseudo_target = np.random.randint(0, 2, size=X.shape[0])

#Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, pseudo_target)

#Use SelectFromModel to select top features based on importance
selector = SelectFromModel(rf, threshold="median", prefit=True)
X_reduced = selector.transform(X)
print(X_reduced.shape)