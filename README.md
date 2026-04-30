# NCERT Assistant RAG Pipeline

This repository implements a modular Retrieval-Augmented Generation (RAG) system specifically designed for processing and querying NCERT textbooks.

## Architecture & Evolution

- **`ncertv2.py`**: This is the foundational script where the complete chunking, embedding, and retrieval process was experimented with and tested first. It contains the raw, step-by-step sequential logic.
- **`rag_pipeline/`**: Once the pipeline was validated, it was structurally framed into a production-ready procedure for the system within the `rag_pipeline` directory. Every component (Loader, Chunking, BM25, Vector Embeddings, Re-ranker) is separated into its own file for maximum maintainability.

---

## How to Operate the Pipeline

The system is designed to split the heavy lifting (parsing, chunking, and embedding) from the querying. This means you do not need to run the entire pipeline again and again for every single query.

### 1. How to Initial Train (Build the Models)
Before you can query the system quickly, you need to parse the PDF and build the vector databases. You only need to do this **once**.

```python
from rag_pipeline.main import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv() # Loads LLAMA_CLOUD_API_KEY from your .env file

pipeline = RAGPipeline()

# This reads the PDF, chunks it, trains the BM25 and FAISS embeddings, 
# and saves everything securely to the "saved_pipeline" folder.
pipeline.build_and_save(pdf_path="ncert-9-1-30.pdf", save_dir="saved_pipeline")
```

### 2. How to Predict (Fast Querying)
Once the embeddings are trained and saved to disk, you can use a single function to get results instantly. It bypasses the LlamaParse API entirely and loads directly from your hard drive.

```python
from rag_pipeline.main import get_results_fast

# Instantly loads from the "saved_pipeline" folder and executes the query
results = get_results_fast("What is the difference between homogeneous and heterogeneous mixtures?")

for score, doc in results:
    print(f"Score: {score:.3f} | Result: {doc.page_content}")
```

---

## Evaluator UI (`retrieval_log.html`)

**`retrieval_log.html`** is the direct replacement for a traditional JSON log file. 

Instead of dumping complex retrieval logs into a messy JSON format, the system generates a beautifully styled static HTML file. This makes it significantly easier for an evaluator to visually assess, read, and compare the different results at every level of the pipeline:
1. **Hybrid Comparison** (Initial retrieval using BM25 + Vectors)
2. **Re-ranker** (Results scored via CrossEncoder)
3. **Sorted with Positive Scores** (Final filtered output)
