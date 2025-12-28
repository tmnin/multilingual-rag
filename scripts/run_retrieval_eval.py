import random
import numpy as np

from noise.spelling import inject_spelling_noise
from retrieval.index import build_faiss_index
from embeddings.cohere_embed import embed_queries

NUM_DOCS = 100
NOISE_PROB = 0.12
TOP_K = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

#dummy set ill replace with wikipedia
def build_dataset():
    docs = []
    for i in range(NUM_DOCS):
        docs.append({
            "doc_id": i,
            "language": "en",
            "clean_text": (
                f"This is a sample document number {i}. "
                f"It contains information about topic {i}."
            )
        })
    return docs

def evaluate():
    docs = build_dataset()

    clean_texts = [d["clean_text"] for d in docs]

    print("Building FAISS index...")
    index = build_faiss_index(clean_texts)

    noisy_queries = [
        inject_spelling_noise(d["clean_text"], prob=NOISE_PROB)
        for d in docs
    ]

    #batched embedding cuz i hit the rate limit
    print("Embedding queries in batch...")
    query_embeddings = embed_queries(noisy_queries)

    query_embeddings = np.array(query_embeddings).astype("float32")
    query_embeddings /= np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )

    recall_at_1 = 0
    recall_at_5 = 0

    for i, doc in enumerate(docs):
        q = query_embeddings[i:i + 1]  # shape (1, dim)

        scores, ids = index.search(q, TOP_K)
        retrieved_ids = ids[0]

        true_id = doc["doc_id"]

        if true_id == retrieved_ids[0]:
            recall_at_1 += 1
        if true_id in retrieved_ids:
            recall_at_5 += 1

    n = len(docs)

    print("\n=== Retrieval Results ===")
    print(f"Documents: {n}")
    print(f"Noise probability: {NOISE_PROB}")
    print(f"Recall@1: {recall_at_1 / n:.3f}")
    print(f"Recall@5: {recall_at_5 / n:.3f}")

if __name__ == "__main__":
    evaluate()
