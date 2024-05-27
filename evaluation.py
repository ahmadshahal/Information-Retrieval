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
        queries_corpus = dict(ir_datasets.load("antique/train").queries_iter()[:100])
    return queries_corpus

    
def __get_qrels_corpus(dataset_name: str):
    if dataset_name == "lifestyle":
        qrels_corpus = list(ir_datasets.load("lotte/lifestyle/dev/search").qrels_iter())
    elif dataset_name == "quora":
        qrels_corpus = list(ir_datasets.load("beir/quora/dev").qrels_iter())
    else:
        normal_qrels_corpus = list(ir_datasets.load("antique/train").qrels_iter())
        qrels_corpus = [item for item in normal_qrels_corpus if item.relevance == 4 or item.relevance == 3]
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

    evaluation_results = ir_measures.calc_aggregate([Success@5, AP@10, RR, P@10, P@5, P@3, R@10], ground_truth, search_results)
    print(evaluation_results)
    

# evaluate("antique")
# {AP@10: 0.06245241522391001, RR: 0.22488798371864888, P@3: 0.11266831547128352, R@10: 0.10870739563803325, P@5: 0.09348722176422052, P@10: 0.06739488870568767, Success@5: 0.302555647155812}

# evaluate("lifestyle")
# {P@5: 0.09784172661870513, P@10: 0.07122302158273397, R@10: 0.26352787234082187, AP@10: 0.14300133837503948, RR: 0.24605632357895402, P@3: 0.10871302957633903, Success@5: 0.35731414868105515}

evaluate("quora")
# {RR: 0.7520686055096847, Success@5: 0.89, P@5: 0.26800000000000035, P@3: 0.35833333333333356, P@10: 0.16150000000000034, R@10: 0.8194242895992895, AP@10: 0.6631131653003081}