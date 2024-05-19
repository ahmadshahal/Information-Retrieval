import pickle

def save_vectorizer(vectorizer, dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/vectorizer_{dataset_name}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

def save_tfidf_matrix(tfidf_matrix, dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/tfidf_matrix_{dataset_name}.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

def get_vectorizer(dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/vectorizer_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

def get_tfidf_matrix(dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/tfidf_matrix_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

def save_means(means, dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/means_{dataset_name}.pkl", "wb") as f:
        pickle.dump(means, f)

def get_means(dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/means_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

def save_clusters(clusters, dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/clusters_{dataset_name}.pkl", "wb") as f:
        pickle.dump(clusters, f)
        
def get_clusters(dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/clusters_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

def save_crawled_dataset(dataset, dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/crawled_{dataset_name}.pkl", "wb") as f:
        pickle.dump(dataset, f)
        
def get_crawled_dataset(dataset_name: str):
    with open(f"/Users/ahmadsmac/ITE/IR/ir-search-engine-master/models/crawled_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

