# importing libraries
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
reducer = UMAP(random_state=42)

import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd
from sklearn.cluster import DBSCAN

#dataset = pd.read_csv(filepath_or_buffer='Use_expression.csv')
#dataset = pd.read_csv(filepath_or_buffer='GSE138852_recon.csv')
#dataset = pd.read_csv(filepath_or_buffer='GSE138852_counts.csv')
dataset = pd.read_csv(filepath_or_buffer='GSE138852_imputed_top700_v2.csv')


dataset_final = dataset.iloc[:, 1:]
data_T = dataset_final.T


standard_embedding = reducer.fit_transform(data_T)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.show()

# dbscan
from sklearn.neighbors import NearestNeighbors
def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):

    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X) 
                                       
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()


k = 2 * data_T.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(X=data_T, k=k)

dbscan_scRNA = DBSCAN(eps=12, min_samples=3).fit(data_T)
dbscan_labels = dbscan_scRNA.labels_
print(np.unique(dbscan_labels))
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=dbscan_labels, s=0.5, cmap='Spectral')
plt.show()

# calculating ARI of dbscan
true_labels_csv = pd.read_csv(filepath_or_buffer='scRNA_true_label_considering_unique_cellType.csv')
true_labels_df = true_labels_csv.iloc[:, 1]
true_labels = true_labels_df.to_numpy(dtype='int64')

dbscan_clustered = (dbscan_labels >= 0)
ARI_dbscan = adjusted_rand_score(dbscan_labels[dbscan_clustered], true_labels[dbscan_clustered])
print(f"ARI_dbscan: {ARI_dbscan}")
print(f"unique values in dbscan_labels: {np.unique(dbscan_labels)}")


# hdbscan
from sklearn.cluster import HDBSCAN
hdbscan_scRNA = HDBSCAN(min_cluster_size=2, min_samples=2).fit(data_T)
hdbscan_labels = hdbscan_scRNA.labels_
print(f"unique values in hdbscan_labels: {np.unique(hdbscan_labels)}")
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=hdbscan_labels, s=0.5, cmap='Spectral')
plt.show()

# calculating ARI of hdbscan
hdbscan_clustered = (hdbscan_labels >= 0)
ARI_hdbscan = adjusted_rand_score(hdbscan_labels[hdbscan_clustered], true_labels[hdbscan_clustered])
print(f"ARI_hdbscan: {ARI_hdbscan}")