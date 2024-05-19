
import ir_datasets
from storage import get_crawled_dataset

def get_corpus(dataset_name: str) -> dict[str, str]:
    """
    Get a corpus of documents for a given dataset name.

    Args:
        dataset_name: The name of the dataset to use. Can be either "lifestyle" or "antique".

    Returns:
        A dictionary mapping document IDs to document content.
    """
    if dataset_name == "lifestyle":

        # TODO: 200,000 documents should be taken from the dataset
        random_corpus = dict(ir_datasets.load("lotte/lifestyle/dev").docs_iter()[:200])
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

    elif dataset_name == "antique":  # dataset_name == "antique":

        # TODO: 200,000 documents should be taken from the dataset
        random_corpus = dict(ir_datasets.load("antique/train").docs_iter()[:200])
        random_corpus_ids = set(random_corpus.keys())

        qrels = list(ir_datasets.load("antique/train").qrels_iter())
        qrels_docs_ids = set(qrel.doc_id for qrel in qrels)

        # Documents that exist in the qrels are also taken in consideration
        docs_ids = random_corpus_ids.union(qrels_docs_ids)

        docs_store = ir_datasets.load("antique/train").docs_store()

        mapped_docs = dict(docs_store.get_many(docs_ids))

        corpus = {doc_id: doc.text for doc_id, doc in mapped_docs.items()}
    
    elif dataset_name == "lifestyle-crawled":
        corpus = get_crawled_dataset("lifestyle")

    elif dataset_name == "antique-crawled":
        corpus = get_crawled_dataset("antique")
    
    elif dataset_name == "lifestyle-queries":
        queries = ir_datasets.load("lotte/lifestyle/dev/forum")
        corpus = dict(queries.queries_iter())
    
    else:
        queries = ir_datasets.load("antique/train")
        corpus = corpus = dict(queries.queries_iter())

    return corpus
