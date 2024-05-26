from text_preprocessing import get_preprocessed_text_terms
from corpus import get_corpus

from storage import save_vectorizer, save_tfidf_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle


def _process_corpus(corpus: dict[str, str]) -> list:
    processed_docs = []
    for doc_key, doc_value in corpus.items():
        processed_doc_terms = get_preprocessed_text_terms(doc_value)
        joined = ' '.join(processed_doc_terms)
        processed_docs.append(joined)
    return processed_docs


def _build_save_vectorizer(dataset_name: str):
    corpus = get_corpus(dataset_name)
    processed_corpus = _process_corpus(corpus)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)

    save_vectorizer(vectorizer, dataset_name)
    save_tfidf_matrix(tfidf_matrix, dataset_name)


# _build_save_vectorizer("lifestyle")
_build_save_vectorizer("quora")
# _build_save_vectorizer("lifestyle-queries")
# _build_save_vectorizer("antique")
# _build_save_vectorizer("antique-queries")
