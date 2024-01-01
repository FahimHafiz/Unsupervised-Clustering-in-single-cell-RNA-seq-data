from umap.umap_ import UMAP
reducer = UMAP(random_state=42)

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Dimension reduction and clustering libraries
import sklearn.cluster as cluster
from sklearn_som.som import SOM
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import pandas as pd

#dataset = pd.read_csv(filepath_or_buffer='Use_expression.csv')
#dataset = pd.read_csv(filepath_or_buffer='GSE138852_recon.csv')
#dataset = pd.read_csv(filepath_or_buffer='GSE138852_counts.csv')
dataset = pd.read_csv(filepath_or_buffer='GSE138852_imputed_top700_v2.csv')


dataset_final = dataset.iloc[:, 1:]
data_T = dataset_final.T


standard_embedding = reducer.fit_transform(data_T)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
plt.show()

som_model = SOM(m=1, n=8, dim=2, random_state=1234)
embedding_12 = standard_embedding[:, 0:2]
som_model.fit(embedding_12)
predictions = som_model.predict(embedding_12)

plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=predictions, s=0.5, cmap='Spectral')
plt.show()

# calculating ARI
true_labels_csv = pd.read_csv(filepath_or_buffer='scRNA_true_label_considering_unique_cellType.csv')
true_labels_df = true_labels_csv.iloc[:, 1]
true_labels = true_labels_df.to_numpy(dtype='int64')

from sklearn.metrics.cluster import adjusted_rand_score

ARI_som = adjusted_rand_score(predictions, true_labels)

print(f"unique clusters: {np.unique(predictions)}")
print(ARI_som)