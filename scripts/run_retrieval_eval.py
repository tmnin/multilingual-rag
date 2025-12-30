import random
import numpy as np

from noise.spelling import apply_spelling_noise
from noise.obfuscation import apply_obfuscation_noise
from noise.chinese import apply_chinese_noise

from retrieval.index import build_faiss_index
from retrieval.baseline_index import build_baseline_faiss_index

from embeddings.cohere_embed import embed_queries
from embeddings.baseline_embed import embed_texts

# -------------------------
# Config
# -------------------------
NUM_DOCS = 40
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]
TOP_K = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Dummy dataset
# -------------------------
def build_dataset():
    docs = []
    for i in range(NUM_DOCS):
        docs.append({
            "doc_id": len(docs),
            "language": "en",
            "clean_text": f"This document discusses topic {i} in English."
        })
        docs.append({
            "doc_id": len(docs),
            "language": "fr",
            "clean_text": f"Ce document traite du sujet {i} en français."
        })
        docs.append({
            "doc_id": len(docs),
            "language": "zh",
            "clean_text": f"这篇文档讨论了主题{i}。"
        })
    return docs


# -------------------------
# Noise pipeline
# -------------------------
def apply_noise(text, language, noise_p):
    if language in ["en", "fr"]:
        text = apply_spelling_noise(text, noise_p)
        text = apply_obfuscation_noise(text, noise_p)
    elif language == "zh":
        text = apply_chinese_noise(text, noise_p)
    return text

# -------------------------
# Evaluation
# -------------------------
def evaluate():
    docs = build_dataset()
    clean_texts = [d["clean_text"] for d in docs]

    print("Building Cohere FAISS index...")
    cohere_index = build_faiss_index(clean_texts)

    print("Building baseline FAISS index...")
    baseline_index = build_baseline_faiss_index(clean_texts)

    print("\n=== Running noise sweep ===")

    results = []

    for noise in NOISE_LEVELS:
        print(f"\nNoise level: {noise}")

        noisy_queries = [
            apply_noise(d["clean_text"], d["language"], noise)
            for d in docs
        ]

        # ---- Cohere embeddings (batched) ----
        cohere_q = embed_queries(noisy_queries)
        cohere_q = np.array(cohere_q).astype("float32")
        cohere_q /= np.linalg.norm(cohere_q, axis=1, keepdims=True)

        # ---- Baseline embeddings ----
        baseline_q = embed_texts(noisy_queries)

        r1_c, r5_c = 0, 0
        r1_b, r5_b = 0, 0

        for i, doc in enumerate(docs):
            true_id = doc["doc_id"]

            # Cohere retrieval
            _, ids_c = cohere_index.search(cohere_q[i:i+1], TOP_K)
            if true_id == ids_c[0][0]:
                r1_c += 1
            if true_id in ids_c[0]:
                r5_c += 1

            # Baseline retrieval
            _, ids_b = baseline_index.search(baseline_q[i:i+1], TOP_K)
            if true_id == ids_b[0][0]:
                r1_b += 1
            if true_id in ids_b[0]:
                r5_b += 1

        n = len(docs)

        results.append({
            "noise": noise,
            "cohere_r1": r1_c / n,
            "cohere_r5": r5_c / n,
            "baseline_r1": r1_b / n,
            "baseline_r5": r5_b / n,
        })

        print(
            f"Cohere   R@1={r1_c/n:.3f} | R@5={r5_c/n:.3f}   ||   "
            f"Baseline R@1={r1_b/n:.3f} | R@5={r5_b/n:.3f}"
        )

    print("\n=== Final Summary ===")
    for r in results:
        print(
            f"Noise={r['noise']:.2f} | "
            f"Cohere R@1={r['cohere_r1']:.3f}, R@5={r['cohere_r5']:.3f} || "
            f"Baseline R@1={r['baseline_r1']:.3f}, R@5={r['baseline_r5']:.3f}"
        )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    evaluate()
