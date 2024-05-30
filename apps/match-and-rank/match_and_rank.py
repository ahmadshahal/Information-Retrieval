
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from query_correction import process_query
from libs.storage import get_tfidf_matrix, get_vectorizer, get_means, get_clusters
from libs.corpus import get_corpus
import requests
from flask import jsonify


def _process_query(query: str, dataset_name: str) -> str:
    response = requests.get(f'http://127.0.0.1:8000/process-text?dataset={dataset_name}&text={query}')
    response.raise_for_status()
    processed_text = response.json()
    return ' '.join(processed_text)


def match_and_rank(query: str, dataset_name: str, similarity_threshold = 0.0001):
    processed_query = _process_query(query, dataset_name)

    loaded_vectorizer = get_vectorizer(dataset_name)
    loaded_tfidf_matrix = get_tfidf_matrix(dataset_name)

    query_vector = loaded_vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(query_vector, loaded_tfidf_matrix)

    corpus = get_corpus(dataset_name)
    doc_ids = corpus.keys()

    document_ranking = dict(zip(doc_ids, similarity_scores.flatten()))

    filtered_documents = {key: value for key, value in document_ranking.items() if value >= similarity_threshold}

    sorted_dict = sorted(filtered_documents.items(), key=lambda item: item[1], reverse=True)

    return OrderedDict(sorted_dict)


# todo: move to clustering section
def clustering_match_and_rank(query: str, dataset_name: str, similarity_threshold = 0.01):
    processed_query = _process_query(query)

    loaded_vectorizer = get_vectorizer(dataset_name)
    loaded_tfidf_matrix = get_tfidf_matrix(dataset_name)

    query_vector = loaded_vectorizer.transform([processed_query])

    k_means = get_means(dataset_name)
    clusters = get_clusters(dataset_name)

    target_cluster = k_means.predict(query_vector)
    cluster_index = [i for i in range(len(clusters)) if(clusters[i]) == target_cluster]

    closest_cluster_vectors = loaded_tfidf_matrix[cluster_index]

    similarity_scores = cosine_similarity(query_vector, closest_cluster_vectors)

    corpus = get_corpus(dataset_name)
    doc_ids = corpus.keys()

    document_ranking = dict(zip(doc_ids, similarity_scores.flatten()))

    filtered_documents = {key: value for key, value in document_ranking.items() if value >= similarity_threshold}

    sorted_dict = sorted(filtered_documents.items(), key=lambda item: item[1], reverse=True)

    return OrderedDict(sorted_dict)
