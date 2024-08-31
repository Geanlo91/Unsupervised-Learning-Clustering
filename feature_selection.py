from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

data = pd.read_csv('final_data.csv')


#Prepare the data
X = data

#Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=5, random_state=42)

#Creating multiple pseudo-targets
importances = np.zeros(X.shape[1])

for _ in range(10):
          pseudo_target = np.random.randint(0, 2, X.shape[0])
          rf.fit(X, pseudo_target)
          importances += rf.feature_importances_

importances /= 10
    

#Use SelectFromModel to select top features based on importance
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_reduced = selector.transform(X.values)
selected_features_SFM = X.columns[selector.get_support()]


#Rank features based on importance using RFE
rfe = RFE(estimator=rf, n_features_to_select=20, step=1)
rfe.fit(X_reduced, pseudo_target)
selected_features_final = selected_features_SFM[rfe.support_]


#visualise the selected features and save them to a csv file
selected_features = pd.DataFrame(selected_features_final, columns=['selected_features'])

#Feature importance visulaization
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()

selected_features_data = X[selected_features_final]
selected_features_data.to_csv('selected_features.csv', index=False)
print(selected_features_data.shape)
print(selected_features_data.head())
