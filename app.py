from flask import Flask, request, jsonify, render_template, send_from_directory
from pathlib import Path
from collections import Counter
import json
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")
client = OpenAI()
app = Flask(__name__, template_folder=BASE_DIR / "templates")

TOP_K = 5

# Load once at startup
with open(BASE_DIR / "data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load(BASE_DIR / "data/embeddings.npy")

print(f"Loaded {len(chunks)} chunks from {BASE_DIR / 'data/chunks.json'}")
print(f"Sources: {sorted(set(c['source'] for c in chunks))}")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def answer_question(query):
    t0 = time.time()

    embed_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(embed_response.data[0].embedding)

    scores = []
    for i, chunk_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, chunks[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scores[:TOP_K]

    context_parts = []
    for i, (score, chunk) in enumerate(top_chunks, start=1):
        header = f"[Source {i}: {chunk['source']}, page {chunk['page']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a scientific research assistant. "
        "Answer the user's question using only the research excerpts provided below. "
        "Be specific and accurate. "
        "Cite your sources inline using the format (Source N: filename, page X). "
        "If the excerpts do not contain enough information to answer the question, say so clearly."
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research excerpts:\n\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )

    latency_ms = round((time.time() - t0) * 1000)
    usage = chat_response.usage
    max_score = top_chunks[0][0] if top_chunks else 0
    low_confidence = bool(max_score < 0.40)

    sources = [
        {
            "label": f"Source {i + 1}",
            "file": chunk["source"],
            "page": chunk["page"],
            "score": round(score, 3),
            "snippet": chunk["text"][:300].strip(),
        }
        for i, (score, chunk) in enumerate(top_chunks)
    ]

    return {
        "answer": chat_response.choices[0].message.content,
        "sources": sources,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "low_confidence": low_confidence,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        result = answer_question(question)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error in /ask")
        return jsonify({"error": str(e)}), 500


@app.route("/papers")
def papers():
    counts = Counter(c["source"] for c in chunks)
    result = sorted(
        [{"file": f, "chunks": n} for f, n in counts.items()],
        key=lambda x: x["file"]
    )
    return jsonify({"papers": result})


@app.route("/document/<path:filename>")
def serve_document(filename):
    return send_from_directory(BASE_DIR / "documents", filename)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port)
