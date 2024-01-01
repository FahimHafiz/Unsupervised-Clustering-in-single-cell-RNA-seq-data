from umap.umap_ import UMAP
reducer = UMAP(random_state=42)

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Dimension reduction and clustering libraries
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd

#dataset = pd.read_csv(filepath_or_buffer='Use_expression.csv')
dataset = pd.read_csv(filepath_or_buffer='GSE138852_recon.csv')

dataset_final = dataset.iloc[:, 1:]
data_T = dataset_final.T


standard_embedding = reducer.fit_transform(data_T)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.show()

kmeans_labels = cluster.KMeans(n_clusters=8).fit_predict(data_T)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.5, cmap='Spectral')
plt.show()

# calculating ARI
true_labels_csv = pd.read_csv(filepath_or_buffer='scRNA_true_label_considering_unique_cellType.csv')
true_labels_df = true_labels_csv.iloc[:, 1]
true_labels = true_labels_df.to_numpy(dtype='int64')

from sklearn.metrics.cluster import adjusted_rand_score

ARI_kmeans = adjusted_rand_score(kmeans_labels, true_labels)