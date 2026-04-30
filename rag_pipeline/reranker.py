from sentence_transformers import CrossEncoder

def rerank_documents(query: str, docs):
    """
    Re-ranks documents using a Cross-Encoder model.
    """
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    candidate_pairs = [(query, doc.page_content) for doc in docs]
    sim_scores = reranker_model.predict(candidate_pairs)
    scored_docs = sorted(zip(sim_scores, docs), key=lambda x: x[0], reverse=True)
    return scored_docs
