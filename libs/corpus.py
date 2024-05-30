import ir_datasets

def get_corpus(dataset_name: str) -> dict[str, str]:
    if dataset_name == "lifestyle":
        corpus = dict(ir_datasets.load("lotte/lifestyle/dev/search").docs_iter())

    elif dataset_name == "antique":
        corpus = dict(ir_datasets.load("antique/train").docs_iter())

    elif dataset_name == "quora":
        corpus = dict(ir_datasets.load("beir/quora/dev").docs_iter())

    elif dataset_name == "lifestyle-crawled":
        corpus = get_crawled_dataset("lifestyle")

    elif dataset_name == "antique-crawled":
        corpus = get_crawled_dataset("antique")

    elif dataset_name == "lifestyle-queries":
        queries = ir_datasets.load("lotte/lifestyle/dev/search")
        corpus = dict(queries.queries_iter())

    else:
        queries = ir_datasets.load("antique/train")
        corpus = corpus = dict(queries.queries_iter())

    return corpus
