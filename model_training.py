import gower
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('selected_features.csv')

#Split the data into train and test
X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)


#compute the Gower distance matrix
gower_dist_train = gower.gower_matrix(X_train)

#Train Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='precomputed')
clusters_train = agg_clustering.fit_predict(gower_dist_train)

#compute the Gower distance matrix for test data
gower_dist_test = gower.gower_matrix(X_test, X_train)

# Assign cluster labels to test data based on the nearest clusters from training data
#The code snippet above demonstrates how to use the Gower distance metric to compute the distance matrix for both training and test data. The training data is then clustered using the Agglomerative Clustering algorithm, and the cluster labels are assigned to the test data based on the nearest clusters from the training data. This approach allows for clustering of test data based on the patterns observed in the training data.
clusters_test = []
for i in range(len(gower_dist_test)):
    min_dist = np.inf
    min_cluster = -1

    for j in range(len(gower_dist_train)):
        dist = gower_dist_test[i][j]

        if dist < min_dist:
            min_dist = dist
            min_cluster = clusters_train[j]
    clusters_test.append(min_cluster)

#Print the cluster labels for test data
print(clusters_test)


#save the cluster labels into a new csv file
cluster_labels = pd.DataFrame(clusters_test, columns=['cluster_label'])
cluster_labels.to_csv('cluster_labels.csv', index=False)

#Calculate the accuracy of the clustering using the silhouette score
sil_score_train = silhouette_score(gower_dist_train, clusters_train, metric='precomputed')
print(f'Silhouette Score (Train): {sil_score_train}')


#group the data by cluster labels
X_train['cluster'] = clusters_train
grouped_data = X_train.groupby('cluster')
print(grouped_data.head())
