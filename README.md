# Multilingual RAG Robustness Under Noisy Queries (Cohere + FAISS)

This project evaluates how a Retrieval-Augmented Generation (RAG) pipeline behaves when user queries are corrupted by multilingual noise (typos / character corruption), compared to a no-retrieval baseline.

The core question: **does retrieval help maintain answer quality and grounding as query noise increases?**

---

## What this project includes

- **Synthetic multilingual QA dataset** (English / French / Chinese):
  - 200 documents total
  - 2 questions per document (400 QA pairs)
  - “Hard negatives”: groups share (org, product, topic) so retrieval is non-trivial
- **Retriever**: Cohere embeddings + FAISS index
- **Generator**: Cohere Chat model
- Noise suite:
  - `spelling` (character-level typos)
  - `obfuscation` (random masking/deletion variants)
  - `chinese` (hanzi perturbations)
  - `codeswitch` (EN<->FR function-word switching and mild English token injection for ZH)
- **Evaluation**:
  - Retrieval: **Hit@5** (whether the correct document is retrieved in the top-5)
  - RAG accuracy: exact/substring match against gold answers
  - Baseline accuracy: same model and same noisy query but without context
  - Faithfulness: whether the gold answer appears in retrieved context (grounding proxy)

---

## Key results (n=60 questions, stratified; corpus=200 docs)

| Noise | Hit@5 | RAG Acc | Baseline Acc | Faithfulness |
|------:|------:|--------:|-------------:|-------------:|
| 0.00  | 0.833 | 0.500   | 0.017        | 0.900        |
| 0.20  | 0.733 | 0.400   | 0.000        | 0.800        |
| 0.40  | 0.567 | 0.350   | 0.000        | 0.650        |

**Interpretation**
- As noise increases, retrieval degrades (Hit@5 drops)
- RAG answer accuracy and faithfulness degrade gradually with retrieval
- The no-RAG baseline collapses under multilingual noise which shows retrieval is necessary for ...

A plot of the noise sweep is available at: `results/rag_noise_sweep_plot.png`.

---

## Setup

```bash
pip install -r requirements.txt
export COHERE_API_KEY="..."
