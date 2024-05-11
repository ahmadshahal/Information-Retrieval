import math
import shelve
from collections import defaultdict
from typing import Dict

import ir_datasets

from text_preprocessing import get_preprocessed_text_terms


def __get_corpus(dataset_name: str) -> Dict[str, str]:
    """
    Get a corpus of documents for a given dataset name.

    Args:
        dataset_name: The name of the dataset to use. Can be either "lifestyle" or "antique".

    Returns:
        A dictionary mapping document IDs to document content.
    """
    if dataset_name == "lifestyle":
        
        # TODO: 200,000 documents should be taken from the dataset
        random_corpus = dict(ir_datasets.load("lotte/lifestyle/dev").docs_iter()[:1000])
        random_corpus_ids = set(random_corpus.keys())

        forum_qrels = list(ir_datasets.load("lotte/lifestyle/dev/forum").qrels_iter())
        search_qrels = list(ir_datasets.load("lotte/lifestyle/dev/search").qrels_iter())

        forum_qrels_docs_ids = set(qrel.doc_id for qrel in forum_qrels)
        search_qrels_docs_ids = set(qrel.doc_id for qrel in search_qrels)

        qrels_docs_ids = forum_qrels_docs_ids.union(search_qrels_docs_ids)

        # Documents that exist in the qrels are also taken in consideration
        docs_ids = random_corpus_ids.union(qrels_docs_ids)

        docs_store = ir_datasets.load("lotte/lifestyle/dev").docs_store()

        mapped_docs = dict(docs_store.get_many(docs_ids))

        corpus = {doc_id: doc.text for doc_id, doc in mapped_docs.items()}

    else:  # dataset_name == "antique":

        # TODO: 200,000 documents should be taken from the dataset
        random_corpus = dict(ir_datasets.load("antique/train").docs_iter()[:1000])
        random_corpus_ids = set(random_corpus.keys())

        qrels = list(ir_datasets.load("antique/train").qrels_iter())
        qrels_docs_ids = set(qrel.doc_id for qrel in qrels)

        # Documents that exist in the qrels are also taken in consideration
        docs_ids = random_corpus_ids.union(qrels_docs_ids)

        docs_store = ir_datasets.load("antique/train").docs_store()

        mapped_docs = dict(docs_store.get_many(docs_ids))

        corpus = {doc_id: doc.text for doc_id, doc in mapped_docs.items()}

    return corpus


def _get_preprocessed_text_terms(text: str, dataset_name: str):
    return get_preprocessed_text_terms(text, dataset_name)


def _calculate_tf(query: str, dataset_name: str) -> Dict[str, float]:
    """
    Calculate the term frequency (TF) for a given query.

    Args:
        query: The query to calculate the TF for.
        dataset_name:The name of the dataset

    Returns:
        A dictionary representing the TF for the given query. The keys are terms and the values are the TF values for
         each term.
    """
    tf = {}
    terms = _get_preprocessed_text_terms(query, dataset_name)
    term_count = len(terms)
    for term in terms:
        tf[term] = terms.count(term) / term_count
    return tf


def _calculate_idf(
    query: str,
    corpus: Dict[str, str],
    dataset_name: str,
    weighted_inverted_index: Dict[str, list],
) -> Dict[str, float]:
    """
    Calculate the inverse document frequency (IDF) for a given corpus and query.

    Returns:
        A dictionary representing the IDF for the given corpus. The keys are terms and the values are the IDF values for
         each term.
    """
    idf = {}
    terms = _get_preprocessed_text_terms(query, dataset_name)
    n_documents = len(corpus)
    for term in terms:
        count = len(weighted_inverted_index[term])
        if(count != 0):
            idf[term] = math.log10(n_documents / count)
        else:
            idf[term] = 0
    return idf


def calculate_query_tfidf(
    query: str,
    dataset_name: str,
    weighted_inverted_index: Dict[str, list],
) -> Dict[str, float]:
    """
    Calculate the TF-IDF for a given query and dataset name.

    Args:
        query: The query to calculate the TF-IDF for.
        dataset_name:The name of the dataset

    Returns:
        A dictionary representing the TF-IDF for the given query. The keys are terms and the values are the TF-IDF
        values for each term.
    """
    tfidf = {}
    tf = _calculate_tf(query, dataset_name)
    corpus = __get_corpus(dataset_name)
    idf = _calculate_idf(query, corpus, dataset_name, weighted_inverted_index)
    for term in tf:
        tfidf[term] = tf[term] * idf.get(term, math.log10(len(corpus) / 1))
        # we need a default value of idf in case of a new term in query(high idf)
    return tfidf


__all__ = [calculate_query_tfidf]
