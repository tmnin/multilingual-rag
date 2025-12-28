import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def embed_documents(texts):
    return co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    ).embeddings

def embed_queries(texts):
    return co.embed(
        texts=texts,
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings