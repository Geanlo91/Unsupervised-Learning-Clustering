from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('final_data.csv')

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=42)

# Train RandomForest on pseudo-targets multiple times
importances = np.zeros(X.shape[1])

for _ in range(20):
    pseudo_target = np.random.randint(0, 2, X.shape[0])
    rf.fit(X, pseudo_target)
    importances += rf.feature_importances_

importances /= 20

# Select top features using SelectFromModel
selector = SelectFromModel(rf, threshold='median', prefit=True)
X_reduced = selector.transform(X)
selected_features_SFM = data.columns[selector.get_support()]

# Rank features based on importance using RFE
rfe = RFE(estimator=rf, n_features_to_select=20, step=1)
rfe.fit(X_reduced, pseudo_target)
print(X_reduced.shape)
selected_features_final = selected_features_SFM[rfe.support_]

# Visualize feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), data.columns[indices], rotation=90)
plt.show()

# Export selected features to CSV
selected_features_data = data[selected_features_final]
selected_features_data.to_csv('selected_features.csv', index=False)

# Print selected features
print(selected_features_data.shape)
print(selected_features_data.head()) 
print(selected_features_data.columns)

