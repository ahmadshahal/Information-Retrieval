from pathlib import Path
import sys
from time import sleep

sys.path.append(str(Path(__file__).parents[2]))

from libs.corpus import get_corpus
from libs import storage
import requests
from flask import jsonify

from sklearn.feature_extraction.text import TfidfVectorizer

def _process_corpus(corpus: dict[str, str], dataset_name: str) -> list:
    processed_docs = []
    for doc_key, doc_value in corpus.items():
        response = requests.get(f'http://127.0.0.1:8000/process-text?dataset={dataset_name}&text={doc_value}')
        response.raise_for_status()
        processed_doc_terms = response.json()

        joined = ' '.join(processed_doc_terms)
        processed_docs.append(joined)
    return processed_docs


def build_save_vectorizer(dataset_name: str):
    corpus = get_corpus(dataset_name)
    processed_corpus = _process_corpus(corpus, dataset_name)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)

    storage.save_vectorizer(vectorizer, dataset_name)
    storage.save_tfidf_matrix(tfidf_matrix, dataset_name)


# build_save_vectorizer("lifestyle")
build_save_vectorizer("lifestyle-queries")
# build_save_vectorizer("antique")
# build_save_vectorizer("quora")
# build_save_vectorizer("antique-queries")

__all__ = ["build_save_vectorizer"]
