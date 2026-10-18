import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

K_VALUES = [1, 3, 5, 10]

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("data/embeddings.npy")

with open("eval_questions.json", "r", encoding="utf-8") as f:
    eval_questions = json.load(f)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query_embedding, k):
    scores = [(cosine_similarity(query_embedding, embeddings[i]), chunks[i])
              for i in range(len(chunks))]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:k]]


# Embed all questions in one API call
print("Embedding evaluation questions...")
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[item["question"] for item in eval_questions]
)
question_embeddings = [np.array(e.embedding) for e in response.data]
print(f"Done — {len(question_embeddings)} questions embedded.\n")

# ── Evaluate ────────────────────────────────────────────────────────
chunk_results = {}   # strict: must match exact chunk ID
source_results = {}  # lenient: correct paper is enough

for K in K_VALUES:
    chunk_p = chunk_r = chunk_mrr = 0
    src_p = src_r = src_mrr = 0

    for i, item in enumerate(eval_questions):
        expected_ids     = set(item["expected_chunk_ids"])
        expected_sources = set(item["expected_sources"])
        retrieved        = semantic_search(question_embeddings[i], K)
        retrieved_ids    = [c["chunk_id"] for c in retrieved]
        retrieved_sources = [c["source"] for c in retrieved]

        # ── Chunk-level (strict) ──
        hit_ids  = [cid for cid in retrieved_ids if cid in expected_ids]
        chunk_p += len(hit_ids) / K
        chunk_r += len(hit_ids) / len(expected_ids)
        ranks = [retrieved_ids.index(cid) + 1 for cid in expected_ids if cid in retrieved_ids]
        chunk_mrr += 1 / min(ranks) if ranks else 0

        # ── Source-level (lenient) ──
        hit_src  = [s for s in retrieved_sources if s in expected_sources]
        src_p   += len(hit_src) / K
        src_r   += 1 if hit_src else 0   # binary: did we find the right paper?
        src_ranks = [retrieved_sources.index(s) + 1 for s in expected_sources if s in retrieved_sources]
        src_mrr  += 1 / min(src_ranks) if src_ranks else 0

    n = len(eval_questions)
    chunk_results[K]  = {"precision": chunk_p/n, "recall": chunk_r/n, "mrr": chunk_mrr/n}
    source_results[K] = {"precision": src_p/n,   "recall": src_r/n,   "mrr": src_mrr/n}

# ── Print results ───────────────────────────────────────────────────
keyword_baseline = {
    1:  {"precision": 0.80, "recall": 0.43, "mrr": 0.80},
    3:  {"precision": 0.33, "recall": 0.50, "mrr": 0.80},
    5:  {"precision": 0.24, "recall": 0.57, "mrr": 0.80},
    10: {"precision": 0.16, "recall": 0.70, "mrr": 0.80},
}

def row(K, d):
    return f"{K:>4}  {d['precision']:>9.2f} {d['recall']:>7.2f} {d['mrr']:>6.2f}"

W = 68
print("=" * W)
print("STRICT — exact chunk match (same metric as keyword baseline)")
print("=" * W)
print(f"{'':4} {'── Keyword baseline ──':^28} {'── Semantic ──':^22}")
print(f"{'K':>4}  {'Precision':>9} {'Recall':>7} {'MRR':>6}   {'Precision':>9} {'Recall':>7} {'MRR':>6}")
print("=" * W)
for K in K_VALUES:
    kb = keyword_baseline[K]
    sm = chunk_results[K]
    print(f"{K:>4}  {kb['precision']:>9.2f} {kb['recall']:>7.2f} {kb['mrr']:>6.2f}"
          f"   {sm['precision']:>9.2f} {sm['recall']:>7.2f} {sm['mrr']:>6.2f}")
print()

print("=" * W)
print("LENIENT — correct paper retrieved (fairer for semantic search)")
print("=" * W)
print(f"{'K':>4}  {'Precision':>9} {'Recall':>7} {'MRR':>6}")
print("-" * W)
for K in K_VALUES:
    print(row(K, source_results[K]))
print()

print("Metrics:")
print("  Precision@K  — of the K chunks retrieved, what fraction were correct?")
print("  Recall@K     — of all correct items, what fraction were found?")
print("  MRR          — how highly ranked was the first correct result?")
print()
print("Note: keyword baseline used literal questions designed for keyword search.")
print("      Semantic evaluation uses natural questions — a fairer real-world test.")
