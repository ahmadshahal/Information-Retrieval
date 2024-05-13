from typing import Dict

import ir_datasets

from ir_measures import *
from match_and_rank import match_and_rank

import ir_measures

def __get_queries_corpus(dataset_name: str) -> Dict[str, str]:
    if dataset_name == "lifestyle":
        queries_corpus = dict(ir_datasets.load("lotte/lifestyle/dev/forum").queries_iter())
    else:
        queries_corpus = dict(ir_datasets.load("antique/train").queries_iter())
    return queries_corpus

    
def __get_qrels_corpus(dataset_name: str):
    if dataset_name == "lifestyle":
        qrels_corpus = list(ir_datasets.load("lotte/lifestyle/dev/forum").qrels_iter())
    else:
        qrels_corpus = list(ir_datasets.load("antique/train").qrels_iter())
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
        results = match_and_rank(query, dataset_name)
        relevance_documents = [(doc_id, score) for doc_id, score in results.items()]
        search_results[query_id] = dict(relevance_documents)
    return search_results


def evaluate(dataset_name: str):
    ground_truth = _get_ground_truth(dataset_name)
    search_results = _get_search_results(dataset_name)

    evaluation_results = ir_measures.calc_aggregate([AP, nDCG, RR, nDCG@10, P@10], ground_truth, search_results)
    print(evaluation_results)


evaluate("antique")
# {P@10: 0.21067600989282878, AP: 0.2058563369194092, RR: 0.5566740539973516, nDCG: 0.3817552297220335, nDCG@10: 0.2921212417779146}

evaluate("lifestyle")
# {P@10: 0.21416184971098332, AP: 0.3184392173592356, RR: 0.5747234047383997, nDCG: 0.514726363643557, nDCG@10: 0.3921174639258026}