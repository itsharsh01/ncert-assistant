import os
import pickle
from .loader import parse_ncert_to_markdown
from .chunking import chunk_ncert_content
from .bm25 import get_bm25_retriever, save_bm25, load_bm25
from .embeddings import get_vector_retriever, save_vector_store, load_vector_retriever
from .comparison import get_hybrid_retriever, compare_without_reranker
from .reranker import rerank_documents
from .sorting import filter_positive_scores

class RAGPipeline:
    def __init__(self):
        self.bm25_retriever = None
        self.vector_retriever = None
        self.hybrid_retriever = None

    def build_and_save(self, pdf_path: str, save_dir: str = "saved_pipeline"):
        """
        Runs the full pipeline (parse, chunk, embed) and saves the models to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        print("1. Parsing PDF...")
        md_content = parse_ncert_to_markdown(pdf_path)
        
        print("2. Chunking...")
        chunks = chunk_ncert_content(md_content)
        
        print("3. Building & Saving BM25...")
        corpus = [chunk.page_content for chunk in chunks]
        self.bm25_retriever = get_bm25_retriever(corpus, k=5)
        save_bm25(self.bm25_retriever, os.path.join(save_dir, "bm25.pkl"))
        
        print("4. Building & Saving Vector Store (Embeddings)...")
        self.vector_retriever, vector_store = get_vector_retriever(chunks, k=5)
        save_vector_store(vector_store, os.path.join(save_dir, "faiss_index"))
        
        self.hybrid_retriever = get_hybrid_retriever(self.bm25_retriever, self.vector_retriever)
        print("Pipeline built and saved successfully!")

    def load_pipeline(self, save_dir: str = "saved_pipeline"):
        """
        Loads the pre-trained embeddings and BM25 from disk instantly.
        """
        self.bm25_retriever = load_bm25(os.path.join(save_dir, "bm25.pkl"))
        self.vector_retriever = load_vector_retriever(os.path.join(save_dir, "faiss_index"), k=5)
        self.hybrid_retriever = get_hybrid_retriever(self.bm25_retriever, self.vector_retriever)

    # --- Query Methods ---
    def query_comparison_without_reranker(self, query: str):
        return compare_without_reranker(self.hybrid_retriever, query)

    def query_with_reranker(self, query: str):
        docs = self.query_comparison_without_reranker(query)
        return rerank_documents(query, docs)

    def query_with_positive_scores_sorting(self, query: str):
        scored_docs = self.query_with_reranker(query)
        return filter_positive_scores(scored_docs, top_k=3)

# -------------------------------------------------------------
# Single Function As Requested
# -------------------------------------------------------------
def get_results_fast(query: str, save_dir="saved_pipeline"):
    """
    A single function to get results instantly without re-running the pipeline.
    Assumes `build_and_save` has already been run once.
    """
    pipeline = RAGPipeline()
    pipeline.load_pipeline(save_dir)
    return pipeline.query_with_positive_scores_sorting(query)
