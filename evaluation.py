from typing import Dict

import ir_datasets

from ir_measures import *
from match_and_rank import match_and_rank, clustering_match_and_rank

import ir_measures


def __get_queries_corpus(dataset_name: str) -> Dict[str, str]:
    if dataset_name == "lifestyle":
        queries_corpus = dict(ir_datasets.load("lotte/lifestyle/dev/search").queries_iter())
    elif dataset_name == "quora":
        queries_corpus = dict(list(ir_datasets.load("beir/quora/dev").queries_iter()))
    else:
        queries_corpus = dict(ir_datasets.load("antique/test").queries_iter())
    return queries_corpus

    
def __get_qrels_corpus(dataset_name: str):
    if dataset_name == "lifestyle":
        qrels_corpus = list(ir_datasets.load("lotte/lifestyle/dev/search").qrels_iter())
    elif dataset_name == "quora":
        qrels_corpus = list(ir_datasets.load("beir/quora/dev").qrels_iter())
    else:
        qrels_corpus = list(ir_datasets.load("antique/test").qrels_iter())
    return qrels_corpus


def _get_ground_truth(dataset_name: str):
    queries_corpus = __get_queries_corpus(dataset_name)
    qrels_corpus = __get_qrels_corpus(dataset_name)
    ground_truth = {}
    for query_id, query in queries_corpus.items():
        relevant_docs = [(item.doc_id, item.relevance) for item in qrels_corpus if item.query_id == query_id]
        ground_truth[query_id] = dict(relevant_docs)
    return ground_truth


def _get_search_results(dataset_name: str):
    search_results = {}
    queries_corpus = __get_queries_corpus(dataset_name)
    for query_id, query in queries_corpus.items():
        print(f'Evaluating query {query_id}')
        results = match_and_rank(query, dataset_name)
        relevance_documents = [(doc_id, score) for doc_id, score in results.items()]
        search_results[query_id] = dict(relevance_documents)
    return search_results


def _get_clustering_search_results(dataset_name: str):
    search_results = {}
    queries_corpus = __get_queries_corpus(dataset_name)
    for query_id, query in queries_corpus.items():
        results = clustering_match_and_rank(query, dataset_name)
        relevance_documents = [(doc_id, score) for doc_id, score in results.items()]
        search_results[query_id] = dict(relevance_documents)
    return search_results


def evaluate(dataset_name: str):
    ground_truth = _get_ground_truth(dataset_name)
    search_results = _get_search_results(dataset_name)
    # search_results = _get_clustering_search_results(dataset_name)

    evaluation_results = ir_measures.calc_aggregate([AP@10, AP, RR, P@10, R@5, R@10], ground_truth, search_results)
    print(evaluation_results)
    

# evaluate("antique")
# {RR: 0.7677160258293952, R@5: 0.082357640324396, AP: 0.2327276313698704, AP@10: 0.1081760933984333, P@10: 0.4035, R@10: 0.13560506468405845}

# evaluate("lifestyle")
# {R@5: 0.23039524424416508, AP: 0.19285465840698576, P@10: 0.07985611510791388, R@10: 0.2868326584153922, AP@10: 0.17339461801684886, RR: 0.31360187349523133}

evaluate("quora")
# {P@5: 0.2030800000000148, R@10: 0.8474052904415849, AP@10: 0.6943854334314166, P@3: 0.3047999999999872, P@10: 0.11340000000000916, Success@10: 0.8908, RR: 0.7378057802071185}