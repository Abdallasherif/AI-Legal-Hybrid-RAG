# -*- coding: utf-8 -*-
"""RAGV0: Hybrid legal RAG pipeline for SCOTUS data.

Originally based on a Colab notebook workflow and adapted into a Python module.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_CASES = 6000
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 350

TOPK_DENSE = 60
TOPK_BM25 = 60
TOPK_FUSED = 80
TOPK_RERANK = 6

ALPHA_DENSE = 0.6
ALPHA_BM25 = 0.4

GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-large"

SYSTEM = """You are a US Supreme Court legal assistant.
Use ONLY the SOURCES.

Output rules:
- If the SOURCES contain relevant information, answer in 2–5 precise sentences.
- Cite each major claim like [S1], [S2].
- Use at most 3 citations total. Do NOT repeat citations.
- Do NOT output "Not found..." if you used any source content.

If the SOURCES do not contain the answer:
- Output exactly: Not found in the provided corpus.
- Output nothing else.

End with <END>.
"""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = clean_text(text)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tok_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [word for word in text.split() if len(word) > 2]


class LegalRAG:
    def __init__(self) -> None:
        self.cases = []
        self.docs = []
        self.bm25 = None
        self.embedder = None
        self.reranker = None
        self.index = None

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
        self.generator = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL,
            quantization_config=bnb,
            device_map="auto",
        )

    def build_corpus(self) -> None:
        ds = load_dataset("lex_glue", "scotus", split="train")

        for row in ds:
            text = row["text"]
            if text and len(text) > 1500:
                self.cases.append(row)
            if len(self.cases) >= MAX_CASES:
                break

        for i, row in enumerate(tqdm(self.cases, desc="Chunking cases")):
            for j, chunk in enumerate(chunk_text(row["text"])):
                self.docs.append({"chunk": chunk, "meta": {"case_id": i, "chunk_id": j}})

        bm25_corpus = [tok_bm25(doc["chunk"]) for doc in self.docs]
        self.bm25 = BM25Okapi(bm25_corpus)

        self.embedder = SentenceTransformer(EMBED_MODEL, device="cuda")
        embeddings = self.embedder.encode(
            [doc["chunk"] for doc in self.docs],
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.reranker = CrossEncoder(RERANK_MODEL, device="cuda")

    def dense_retrieve(self, query: str, k: int = TOPK_DENSE) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return list(zip(idx[0].tolist(), scores[0].tolist()))

    def bm25_retrieve(self, query: str, k: int = TOPK_BM25) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(tok_bm25(query))
        top = np.argsort(scores)[::-1][:k]
        return list(zip(top.tolist(), scores[top].tolist()))

    @staticmethod
    def fuse(
        dense: Sequence[Tuple[int, float]],
        sparse: Sequence[Tuple[int, float]],
        a: float = ALPHA_DENSE,
        b: float = ALPHA_BM25,
        k: int = TOPK_FUSED,
    ) -> List[Tuple[int, float]]:
        def norm(items: Sequence[Tuple[int, float]]) -> Dict[int, float]:
            if not items:
                return {}
            scores = np.array([x[1] for x in items], dtype="float32")
            if np.max(scores) - np.min(scores) < 1e-6:
                scores = np.ones_like(scores)
            else:
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            return {items[i][0]: float(scores[i]) for i in range(len(items))}

        dense_map, sparse_map = norm(dense), norm(sparse)
        all_ids = set(dense_map) | set(sparse_map)
        fused = [(i, a * dense_map.get(i, 0.0) + b * sparse_map.get(i, 0.0)) for i in all_ids]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]

    def rerank(self, query: str, fused_results: Sequence[Tuple[int, float]], topk: int = TOPK_RERANK):
        pairs = [(query, self.docs[i]["chunk"]) for i, _ in fused_results]
        scores = self.reranker.predict(pairs, batch_size=64)
        scored = [(fused_results[i][0], float(scores[i])) for i in range(len(fused_results))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def format_sources(self, reranked: Sequence[Tuple[int, float]], max_sources: int | None = None) -> str:
        max_sources = max_sources or len(reranked)
        blocks = []
        for r, (idx, _) in enumerate(reranked[:max_sources], 1):
            m = self.docs[idx]["meta"]
            head = f"[S{r}] case_id={m['case_id']} chunk={m['chunk_id']}"
            snippet = self.docs[idx]["chunk"][:900]
            blocks.append(head + "\n" + snippet)
        return "\n\n".join(blocks)

    def answer(self, query: str) -> Tuple[str, List[Tuple[int, float]]]:
        dense = self.dense_retrieve(query, TOPK_DENSE)
        sparse = self.bm25_retrieve(query, TOPK_BM25)
        fused = self.fuse(dense, sparse, ALPHA_DENSE, ALPHA_BM25, TOPK_FUSED)
        reranked = self.rerank(query, fused, TOPK_RERANK)

        if reranked and reranked[0][1] < 0.75:
            return "Not found in the provided corpus.", reranked

        sources = self.format_sources(reranked, max_sources=min(TOPK_RERANK, 6))
        prompt = (
            SYSTEM
            + "\n\nQUESTION:\n"
            + query
            + "\n\nSOURCES:\n"
            + sources
            + "\n\nWrite the ANSWER only. End your answer with the token <END>.\n\nANSWER:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(self.generator.device)
        with torch.no_grad():
            output = self.generator.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                use_cache=True,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "ANSWER:" in text:
            text = text.split("ANSWER:", 1)[-1]
        if "<END>" in text:
            text = text.split("<END>", 1)[0]

        return text.strip(), reranked

    def retrieve_ranked_indices(self, query: str) -> List[int]:
        dense = self.dense_retrieve(query, TOPK_DENSE)
        sparse = self.bm25_retrieve(query, TOPK_BM25)
        fused = self.fuse(dense, sparse, ALPHA_DENSE, ALPHA_BM25, TOPK_FUSED)
        reranked = self.rerank(query, fused, TOPK_RERANK)
        return [idx for idx, _ in reranked]



    def launch_gradio(self):
        import gradio as gr

        def chat_fn(question):
            response, _ = self.answer(question)
            return response

        demo = gr.Interface(
            fn=chat_fn,
            inputs=gr.Textbox(lines=2, placeholder="Ask Elite..."),
            outputs="text",
            title="AI Legal RAG Assistant",
            description="Hybrid Retrieval (BM25 + BGE-M3) + Cross-Encoder Reranking + Grounded Generation",
            examples=[
                "probable cause warrant requirement",
                "prior restraint first amendment",
                "double jeopardy blockburger test",
                "miller test obscenity",
                "qualified immunity clearly established law",
            ],
        )
        demo.launch(share=True)

    def eval_retrieval(self, num_samples: int = 200, k_list: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
        def make_query_from_case(text: str):
            return clean_text(text)[:250]

        def recall_at_k(ranked_chunk_ids, target_case_id, k):
            topk = ranked_chunk_ids[:k]
            return 1.0 if any(self.docs[i]["meta"]["case_id"] == target_case_id for i in topk) else 0.0

        def mrr_at_k(ranked_chunk_ids, target_case_id, k):
            topk = ranked_chunk_ids[:k]
            for rank, cid in enumerate(topk, start=1):
                if self.docs[cid]["meta"]["case_id"] == target_case_id:
                    return 1.0 / rank
            return 0.0

        idxs = random.sample(range(len(self.cases)), min(num_samples, len(self.cases)))
        results = {f"Recall@{k}": [] for k in k_list}
        results.update({f"MRR@{k}": [] for k in k_list})

        for case_id in tqdm(idxs, desc="Evaluating retrieval"):
            query = make_query_from_case(self.cases[case_id]["text"])
            ranked = self.retrieve_ranked_indices(query)
            for k in k_list:
                results[f"Recall@{k}"].append(recall_at_k(ranked, case_id, k))
                results[f"MRR@{k}"].append(mrr_at_k(ranked, case_id, k))

        return {metric: float(np.mean(values)) for metric, values in results.items()}


if __name__ == "__main__":
    print("FAISS version:", faiss.__version__)
    rag = LegalRAG()
    rag.build_corpus()
    response, ranked = rag.answer("miranda custodial interrogation")
    print(response[:500])
    if ranked:
        print("Top rerank score:", ranked[0][1])
