import numpy as np
from embeddings.cohere_embed import embed_queries

def search(index, query, k=5):
    q_emb = embed_queries([query])
    q = np.array(q_emb)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    scores, ids = index.search(q, k)
    return ids[0], scores[0]
