# Multilingual RAG Robustness Benchmark (Cohere + FAISS)

This project benchmarks how Retrieval-Augmented Generation (RAG) holds up when user queries are **noisy and multilingual**. It evaluates retrieval quality and answer quality across **English / French / Chinese** under controlled perturbations (spelling noise, obfuscation, and code-switching).

The goal is to build a small but realistic evaluation harness that demonstrates:
- how retrieval degrades under noise (Hit@K)
- how RAG answer accuracy tracks retrieval quality
- how often answers remain grounded in retrieved context (faithfulness)

## What this project includes

- **Dataset generator**: produces a 200-document multilingual corpus with hard negatives (documents sharing org/product/topic but differing in key facts).
- **Noise pipeline**:
  - spelling corruption
  - lightweight obfuscation
  - **code-switching** (EN↔FR function-word swapping; occasional English injection for ZH)
- **Retrieval**: Cohere embeddings + FAISS index (L2-normalized vectors)
- **RAG**: Cohere Chat with a strict prompt (“use ONLY the provided context”)
- **Metrics**:
  - Retrieval Hit@K
  - Answer accuracy (string match / normalized)
  - Faithfulness (gold answer present in retrieved contexts)

## Key results (n=60 questions, stratified; corpus=200 docs)

| Noise | Hit@5 | RAG Acc | Baseline Acc | Faithfulness |
|------:|------:|--------:|-------------:|-------------:|
| 0.00  | 0.833 | 0.500   | 0.017        | 0.900        |
| 0.20  | 0.733 | 0.400   | 0.000        | 0.800        |
| 0.40  | 0.567 | 0.350   | 0.000        | 0.650        |

As noise increases, retrieval degrades and RAG accuracy declines more gradually than the no-RAG baseline.
*(Exact numbers vary slightly depending on the evaluation subset and random seed.)*

## Notes
- Trial keys are rate-limited. The evaluation code throttles chat calls to avoid 429 errors.
- Faithfulness here is measured as a controlled “gold-in-context” check (appropriate for this synthetic corpus).

## Future work
- Add stronger multilingual noise (script conversion, transliteration, OCR-like errors)
- Evaluate alternate retrievers / rerankers
- Replace exact-match answer accuracy with a semantic evaluator (LLM judge or similarity model)

## Setup

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set Cohere API key
```bash
export COHERE_API_KEY="..."
```



