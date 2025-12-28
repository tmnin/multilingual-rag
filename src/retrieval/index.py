import faiss
import numpy as np
from embeddings.cohere_embed import embed_documents

def build_faiss_index(texts):
    embeddings = embed_documents(texts)

    xb = np.array(embeddings).astype("float32") #covert to float32

    dim = xb.shape[1]

    faiss.normalize_L2(xb) #normalize 

    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    return index
