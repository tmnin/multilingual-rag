import random
import numpy as np

from noise.spelling import inject_spelling_noise
from noise.obfuscation import inject_obfuscation
from retrieval.index import build_faiss_index
from embeddings.cohere_embed import embed_queries

# -------------------------
# Config
# -------------------------
NUM_DOCS = 100
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]
TOP_K = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Dummy dataset
# (replace later with Wikipedia)
# -------------------------
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

# -------------------------
# Noise pipeline
# -------------------------
def apply_noise(text, spelling_p=0.1, obfuscation_p=0.1):
    text = inject_spelling_noise(text, prob=spelling_p)
    text = inject_obfuscation(text, prob=obfuscation_p)
    return text

# -------------------------
# Evaluation
# -------------------------
def evaluate():
    docs = build_dataset()
    clean_texts = [d["clean_text"] for d in docs]

    # -------------------------------------------------
    # Build FAISS index on CLEAN documents
    # -------------------------------------------------
    print("Building FAISS index...")
    index = build_faiss_index(clean_texts)

    results = []

    # -------------------------------------------------
    # Noise sweep
    # -------------------------------------------------
    for noise in NOISE_LEVELS:
        print(f"\nEvaluating noise level: {noise}")

        noisy_queries = [
            apply_noise(
                d["clean_text"],
                spelling_p=noise,
                obfuscation_p=noise
            )
            for d in docs
        ]

        print("Embedding queries in batch...")
        query_embeddings = embed_queries(noisy_queries)
        query_embeddings = np.array(query_embeddings).astype("float32")
        query_embeddings /= np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )

        recall_at_1 = 0
        recall_at_5 = 0

        for i, doc in enumerate(docs):
            q = query_embeddings[i:i + 1]
            scores, ids = index.search(q, TOP_K)
            retrieved = ids[0]

            if doc["doc_id"] == retrieved[0]:
                recall_at_1 += 1
            if doc["doc_id"] in retrieved:
                recall_at_5 += 1

        n = len(docs)
        r1 = recall_at_1 / n
        r5 = recall_at_5 / n

        results.append({
            "noise": noise,
            "recall@1": r1,
            "recall@5": r5
        })

        print(f"Recall@1: {r1:.3f} | Recall@5: {r5:.3f}")

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------
    print("\n=== Summary ===")
    for r in results:
        print(
            f"Noise={r['noise']:.2f} | "
            f"R@1={r['recall@1']:.3f} | "
            f"R@5={r['recall@5']:.3f}"
        )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    evaluate()
