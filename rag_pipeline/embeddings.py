import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_vector_retriever(chunks, k=5):
    """
    Initializes and returns a Vector Store Retriever.
    """
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return vector_retriever, vector_store

def save_vector_store(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vector_retriever(path="faiss_index", k=5):
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization=True is needed for local FAISS loading
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": k})
