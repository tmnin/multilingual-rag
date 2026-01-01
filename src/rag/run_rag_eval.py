import json
import os
import time
import random
from typing import List, Tuple, Optional, Set, Dict, Any

from tqdm import tqdm
import cohere
from cohere.errors import TooManyRequestsError

from retrieval.index import build_faiss_index
from retrieval.search import search
from noise.apply_noise import apply_noise
from rag.metrics import is_correct, normalize


# --------------------
# Config
# --------------------

DATA_PATH = "data/qa_dataset_200.json"
TOP_K = 5

NOISE_LEVELS = [0.0, 0.2, 0.4]

COHERE_MODEL = "command-r7b-12-2024"

MIN_SECONDS_BETWEEN_CALLS = 3.6

MAX_EXAMPLES: Optional[int] = 60
EVAL_LANGS: Set[str] = {"en", "fr", "zh"}
RANDOM_SEED = 42

SAVE_RESULTS = True
RESULTS_DIR = "results"
RUN_NAME = f"rag_eval_{int(time.time())}"

co = cohere.Client(os.environ.get("COHERE_API_KEY"))

_last_call_ts = 0.0


# --------------------
# Cohere wrapper (throttle + infinite retry)
# --------------------

def chat_with_backoff(**kwargs):
    """
    - Global throttling to respect trial rate limits
    - On 429: keep sleeping and retrying (do NOT crash)
    """
    global _last_call_ts

    # throttle before each call
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_CALLS - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    backoff = 2.0
    while True:
        try:
            resp = co.chat(**kwargs)
            _last_call_ts = time.time()
            return resp
        except TooManyRequestsError:
            # back off and try again
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 45.0)


# --------------------
# Generation
# --------------------

def generate_answer_rag(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)

    prompt = f"""Answer the question using ONLY the context below.
If the answer cannot be found in the context, say "I don't know".

Context:
{context_block}

Question:
{query}

Answer:"""

    response = chat_with_backoff(
        model=COHERE_MODEL,
        message=prompt,
        temperature=0.0,
        max_tokens=120,
    )
    return response.text.strip()


def generate_answer_no_rag(query: str) -> str:
    prompt = f"""Answer the question. If you are unsure, say "I don't know".

Question:
{query}

Answer:"""

    response = chat_with_backoff(
        model=COHERE_MODEL,
        message=prompt,
        temperature=0.0,
        max_tokens=120,
    )
    return response.text.strip()


# --------------------
# Metrics
# --------------------

def is_faithful_gold(gold: str, contexts: List[str]) -> bool:
    gold_n = normalize(gold)
    ctx_n = " ".join(normalize(c) for c in contexts)
    return gold_n in ctx_n


# --------------------
# Data helpers
# --------------------

def flatten_dataset(data: list) -> Tuple[List[str], List[dict]]:
    documents: List[str] = []
    examples: List[dict] = []

    for doc_index, item in enumerate(data):
        documents.append(item["document"])
        lang = item.get("language", "en")
        for q in item.get("questions", []):
            examples.append(
                {
                    "doc_id": item.get("doc_id", str(doc_index)),
                    "qid": q.get("qid", f"{item.get('doc_id', doc_index)}_q"),
                    "language": lang,
                    "question": q["question"],
                    "answer": q["answer"],
                    "true_doc_index": doc_index,
                }
            )
    return documents, examples


def stratified_sample(examples: List[dict], max_examples: int, langs: Set[str], seed: int) -> List[dict]:
    rng = random.Random(seed)
    filtered = [ex for ex in examples if ex["language"] in langs]
    if len(filtered) <= max_examples:
        return filtered

    per_lang = max_examples // len(langs)
    sampled: List[dict] = []

    for lang in sorted(langs):
        lang_ex = [ex for ex in filtered if ex["language"] == lang]
        rng.shuffle(lang_ex)
        sampled.extend(lang_ex[:per_lang])

    if len(sampled) < max_examples:
        remaining = [ex for ex in filtered if ex not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: max_examples - len(sampled)])

    return sampled


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    """
    Make sure we never crash json.dumps on numpy arrays, etc.
    We also avoid saving big stuff by design.
    """
    try:
        import numpy as np  

        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# --------------------
# Main evaluation
# --------------------

def run_rag_eval():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents, examples = flatten_dataset(data)

    if MAX_EXAMPLES is not None:
        examples = stratified_sample(examples, MAX_EXAMPLES, EVAL_LANGS, RANDOM_SEED)

    print(f"Index docs: {len(documents)} | Eval questions: {len(examples)}")
    print("Building FAISS index...")
    index = build_faiss_index(documents)

    if SAVE_RESULTS:
        ensure_dir(RESULTS_DIR)
        jsonl_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}.jsonl")
        summary_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}_summary.json")
    else:
        jsonl_path = None
        summary_path = None

    results_summary: List[Dict[str, float]] = []

    jsonl_f = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None

    try:
        for noise_p in NOISE_LEVELS:
            print(f"\nEvaluating noise level: {noise_p}")

            total = 0
            rag_correct = 0
            base_correct = 0
            faithful = 0
            retrieval_hit = 0

            for ex in tqdm(examples):
                total += 1

                noisy_q = apply_noise(
                    ex["question"],
                    language=ex["language"],
                    noise_p=noise_p,
                )

                retrieved_ids, _ = search(index, noisy_q, k=TOP_K)
                retrieved_contexts = [documents[i] for i in retrieved_ids]

                if ex["true_doc_index"] in retrieved_ids:
                    retrieval_hit += 1

                rag_answer = generate_answer_rag(noisy_q, retrieved_contexts)
                base_answer = generate_answer_no_rag(noisy_q)

                gold = ex["answer"]

                rag_ok = is_correct(rag_answer, gold)
                base_ok = is_correct(base_answer, gold)
                faithful_ok = is_faithful_gold(gold, retrieved_contexts)

                if rag_ok:
                    rag_correct += 1
                if base_ok:
                    base_correct += 1
                if faithful_ok:
                    faithful += 1

                if jsonl_f:
                    row = {
                        "run_name": RUN_NAME,
                        "noise": noise_p,
                        "qid": ex["qid"],
                        "doc_id": ex["doc_id"],
                        "language": ex["language"],
                        "question": ex["question"],
                        "noisy_question": noisy_q,
                        "gold": gold,
                        "retrieved_ids": retrieved_ids, 
                        "true_doc_index": ex["true_doc_index"],
                        "hit@k": ex["true_doc_index"] in retrieved_ids,
                        "rag_answer": rag_answer,
                        "base_answer": base_answer,
                        "rag_correct": rag_ok,
                        "base_correct": base_ok,
                        "faithful_gold_in_context": faithful_ok,
                    }
                    row = {k: to_jsonable(v) for k, v in row.items()}
                    jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            hitk = retrieval_hit / total
            rag_acc = rag_correct / total
            base_acc = base_correct / total
            faith = faithful / total

            print(f"Retrieval Hit@{TOP_K}: {hitk:.3f}")
            print(f"RAG Accuracy:      {rag_acc:.3f}")
            print(f"Baseline Accuracy: {base_acc:.3f}")
            print(f"Faithfulness:      {faith:.3f}")

            results_summary.append(
                {
                    "noise": noise_p,
                    f"hit@{TOP_K}": hitk,
                    "rag_acc": rag_acc,
                    "base_acc": base_acc,
                    "faithfulness": faith,
                }
            )

        print("\n=== Final RAG Summary ===")
        for r in results_summary:
            print(
                f"Noise={r['noise']:.2f} | "
                f"Hit@{TOP_K}={r[f'hit@{TOP_K}']:.3f} | "
                f"RAG Acc={r['rag_acc']:.3f} | "
                f"Base Acc={r['base_acc']:.3f} | "
                f"Faithful={r['faithfulness']:.3f}"
            )

        if summary_path:
            payload = {
                "run_name": RUN_NAME,
                "data_path": DATA_PATH,
                "top_k": TOP_K,
                "noise_levels": NOISE_LEVELS,
                "max_examples": MAX_EXAMPLES,
                "eval_langs": sorted(list(EVAL_LANGS)),
                "model": COHERE_MODEL,
                "min_seconds_between_calls": MIN_SECONDS_BETWEEN_CALLS,
                "summary": results_summary,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"\nSaved per-example results: {jsonl_path}")
            print(f"Saved summary:            {summary_path}")

    finally:
        if jsonl_f:
            jsonl_f.close()


if __name__ == "__main__":
    run_rag_eval()
