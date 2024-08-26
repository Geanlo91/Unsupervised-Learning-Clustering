from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv('final_data.csv')


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
X_reduced = selector.transform(X.values)
selected_features = X.columns[selector.get_support()]
#print(X_reduced.shape)
#print(selected_features)

#save the selected features into a new csv file
X_selected = pd.DataFrame(X_reduced, columns=selected_features)
X_selected.to_csv('selected_features.csv', index=False)
#print(X_selected.dtypes)
print(X_selected.shape)


# Splits the data into training and test sets.
# Computes the Gower distance matrix for the training data.
# Trains an Agglomerative Clustering model on the training data.
# Computes the Gower distance matrix for the test data with respect to the training data.
# Assigns cluster labels to the test data based on the nearest clusters from the training data.
# Saves the cluster labels for the test data into a new CSV file.
# This function can be used to cluster new data based on the patterns observed in the training data, allowing for the application of machine learning models to new datasets.
# The code snippet above demonstrates how to use the Gower distance metric to compute the distance matrix for both training and test data. The training data is then clustered using the Agglomerative Clustering algorithm, and the cluster labels are assigned to the test data based on the nearest clusters from the training data. This approach allows for clustering of test data based on the patterns observed in the training data.

#alculate the silhouette score using the precomputed Gower distance matrix