# ⚖️📚 AI Legal RAG Assistant

A hybrid **Retrieval-Augmented Generation (RAG)** pipeline for U.S. Supreme Court legal research using:

- 🔎 **Dense retrieval** with `BAAI/bge-m3`
- 🧾 **Sparse retrieval** with BM25
- 🧠 **Cross-encoder reranking** with `BAAI/bge-reranker-large`
- ✍️ **Grounded answer generation** with `Qwen/Qwen2.5-7B-Instruct`
- ⚡ **FAISS** for fast vector search

---

## ✨ Features

- 📦 Loads SCOTUS data from `lex_glue`
- ✂️ Splits long cases into overlapping chunks
- 🔀 Fuses dense + sparse retrieval scores
- 🥇 Reranks top candidates for higher relevance
- 🧪 Includes retrieval evaluation with Recall@K and MRR@K
- 🛡️ Returns `Not found in the provided corpus.` when confidence is low

---

## 🧰 Requirements

- Python 3.10+
- CUDA-capable GPU recommended (for embedding, reranking, and generation)

Install dependencies:

```bash
pip install -U datasets transformers accelerate bitsandbytes sentence-transformers faiss-cpu rank-bm25 tqdm
```

---

## 🚀 Quick Start

```bash
python RAGV0.py
```

This will:
1. Load SCOTUS training data
2. Build BM25 + FAISS indices
3. Run a sample query: `miranda custodial interrogation`

---

## 🧪 Programmatic Usage

```python
from RAGV0 import LegalRAG

rag = LegalRAG()
rag.build_corpus()

answer, reranked = rag.answer("prior restraint first amendment")
print(answer)
print(reranked[:3])

metrics = rag.eval_retrieval(num_samples=200, k_list=(1, 3, 5, 10))
print(metrics)
```

---

## 📝 Notes

- 💡 The script is adapted from a Colab-style notebook workflow.
- 🧱 Model downloads are large and may take time on first run.
- 🖥️ Running on CPU is possible but very slow.

---

## 📄 License

Use this repository in accordance with the licenses of all upstream models and datasets.
