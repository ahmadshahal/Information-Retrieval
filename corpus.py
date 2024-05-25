
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
        corpus = dict(ir_datasets.load("lotte/lifestyle/dev/forum").docs_iter())

    elif dataset_name == "antique":
        corpus = dict(ir_datasets.load("antique/train").docs_iter())
    
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
