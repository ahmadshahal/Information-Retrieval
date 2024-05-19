from storage import get_tfidf_matrix, save_clusters, save_means
from sklearn.cluster import KMeans
import numpy as np

def build_save_cluster(dataset_name):
    loaded_tfidf_matrix = get_tfidf_matrix(dataset_name)

    k_means = KMeans(n_clusters=25)
    clusters = k_means.fit_predict(loaded_tfidf_matrix)

    counts = np.bincount(clusters)
    for i, count in enumerate(counts):
        print(f'Cluster {i} has {count} vectors')

    save_means(k_means, dataset_name=dataset_name)
    save_clusters(clusters=clusters, dataset_name=dataset_name)

build_save_cluster("lifestyle")
# build_save_cluster("antique")