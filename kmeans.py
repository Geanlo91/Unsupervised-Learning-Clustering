from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv('selected_features.csv') 

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    return clusters

#determine the number of clusters
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal number of Clusters')    
plt.show()

#Train KMeans Clustering
clusters_train = kmeans_clustering(data, n_clusters=2)

#Assign cluster labels to original data
data['cluster'] = clusters_train


#Calculate the accuracy of the clustering using the silhouette score
sil_score_train = silhouette_score(data.drop(columns=['cluster']), clusters_train)
print(f'Silhouette Score (Train): {sil_score_train}')

#group the data by cluster labels
data['cluster'] = clusters_train
grouped_data = data.groupby('cluster')


#Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data.drop(columns=['cluster']))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_train, cmap='viridis')
plt.title('PCA of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

#save the data with cluster labels into a new csv file
data.to_csv('clustered_data.csv', index=False)









