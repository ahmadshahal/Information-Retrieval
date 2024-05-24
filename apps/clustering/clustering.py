from libs import storage
from sklearn.cluster import KMeans
import numpy as np

def build_save_cluster(dataset_name):
    loaded_tfidf_matrix = storage.get_tfidf_matrix(dataset_name)

    k_means = KMeans(n_clusters=25)
    clusters = k_means.fit_predict(loaded_tfidf_matrix)

    counts = np.bincount(clusters)
    for i, count in enumerate(counts):
        print(f'Cluster {i} has {count} vectors')

    storage.save_means(k_means, dataset_name=dataset_name)
    storage.save_clusters(clusters=clusters, dataset_name=dataset_name)

# build_save_cluster("lifestyle")
# build_save_cluster("antique")
