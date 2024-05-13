import pickle

def save_vectorizer(vectorizer, dataset_name: str):
    with open(f"vectorizer_{dataset_name}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

def save_tfidf_matrix(tfidf_matrix, dataset_name: str):
    with open(f"tfidf_matrix_{dataset_name}.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

def get_vectorizer(dataset_name: str):
    with open(f"vectorizer_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

def get_tfidf_matrix(dataset_name: str):
    with open(f"tfidf_matrix_{dataset_name}.pkl", "rb") as f:
        return pickle.load(f)

