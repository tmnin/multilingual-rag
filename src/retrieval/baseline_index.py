import faiss
import numpy as np
from embeddings.baseline_embed import embed_texts

def build_baseline_faiss_index(texts):
    xb = embed_texts(texts)
    dim = xb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    return index
