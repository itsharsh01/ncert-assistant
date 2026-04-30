import pickle
from langchain_community.retrievers import BM25Retriever

def get_bm25_retriever(corpus_texts, k=5):
    """
    Initializes and returns a BM25 Retriever.
    """
    bm25_retriever = BM25Retriever.from_texts(corpus_texts)
    bm25_retriever.k = k
    return bm25_retriever

def save_bm25(retriever, path="bm25.pkl"):
    with open(path, "wb") as f:
        pickle.dump(retriever, f)

def load_bm25(path="bm25.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
