from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(bm25_retriever, vector_retriever, bm25_weight=0.4, vector_weight=0.6):
    """
    Combines BM25 and Vector retrievers into a hybrid retriever.
    """
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    return hybrid_retriever

def compare_without_reranker(hybrid_retriever, query: str):
    """
    Retrieves chunks based on the query using hybrid retrieval.
    This serves as comparison without reranker.
    """
    final_docs = hybrid_retriever.invoke(query)
    return final_docs
