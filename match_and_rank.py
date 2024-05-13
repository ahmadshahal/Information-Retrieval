import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

from storage import get_tfidf_matrix, get_vectorizer
from corpus import get_corpus

from text_preprocessing import get_preprocessed_text_terms

def _process_query(query: str) -> str:
    processed_query = get_preprocessed_text_terms(query)
    return ' '.join(processed_query)


def match_and_rank(query: str, dataset_name: str, similarity_threshold = 0.001):
    processed_query = _process_query(query)

    loaded_vectorizer = get_vectorizer(dataset_name)
    loaded_tfidf_matrix = get_tfidf_matrix(dataset_name)
    
    query_vector = loaded_vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(query_vector, loaded_tfidf_matrix)

    corpus = get_corpus(dataset_name)
    doc_ids = corpus.keys()

    filtered_similarities = [score for score in similarity_scores.flatten() if score >= similarity_threshold]

    document_ranking = dict(zip(doc_ids, filtered_similarities))
    sorted_dict = sorted(document_ranking.items(), key=lambda item: item[1], reverse=True)

    return OrderedDict(sorted_dict)
