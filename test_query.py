import os
from rag_pipeline.main import RAGPipeline

# 1. Load your LlamaParse API Key from the .env file
from dotenv import load_dotenv
load_dotenv()

# 2. Initialize the pipeline with your PDF file
# This will automatically parse, chunk, and create embeddings for the document
pipeline = RAGPipeline(pdf_path="ncert-9-1-30.pdf")

# 3. Define the query you want to ask
my_query = "What is the difference between homogeneous and heterogeneous mixtures?"

# ---------------------------------------------------------
# METHOD 1: Comparison Without Re-ranker
# Uses the Hybrid Retriever (BM25 + Vector Embeddings) only
# ---------------------------------------------------------
print("--- 1. Comparison Without Re-ranker ---")
results_1 = pipeline.query_comparison_without_reranker(my_query)
for i, doc in enumerate(results_1, 1):
    print(f"Result {i}: {doc.page_content}\n")

# ---------------------------------------------------------
# METHOD 2: Query With Re-ranker
# Uses Hybrid Retriever + CrossEncoder Re-ranking
# ---------------------------------------------------------
print("--- 2. With Re-ranker ---")
results_2 = pipeline.query_with_reranker(my_query)
for i, (score, doc) in enumerate(results_2, 1):
    print(f"Result {i} (Score: {score:.4f}): {doc.page_content}\n")

# ---------------------------------------------------------
# METHOD 3: Sorted With Positive Scores
# Filters the Re-ranked results to only return positive scores (top 3)
# ---------------------------------------------------------
print("--- 3. Sorted With Positive Scores ---")
results_3 = pipeline.query_with_positive_scores_sorting(my_query)
for i, (score, doc) in enumerate(results_3, 1):
    print(f"Result {i} (Score: {score:.4f}): {doc.page_content}\n")
