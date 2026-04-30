def filter_positive_scores(scored_docs, top_k=3):
    """
    Filters for documents with positive scores and returns the top k.
    """
    filtered_docs = []
    for score, doc in scored_docs:
        if score > 0:
            filtered_docs.append((score, doc))
        if len(filtered_docs) >= top_k:
            break
    return filtered_docs
