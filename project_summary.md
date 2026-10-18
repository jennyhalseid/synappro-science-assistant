# synappro Science RAG

## Project idea

This project is a RAG (Retrieval-Augmented Generation) system for synappro that retrieves scientific evidence from research papers about breathing, stress, HRV, mindfulness and the nervous system.

The goal is to help synappro make evidence-based claims in website copy, blogs, social media posts, workshops and grant applications.

## Current state

A fully working semantic RAG system with a web UI, ready for deployment.

### Pipeline

1. **`chunk_documents.py`** — reads all PDFs in `documents/`, splits each page into overlapping text chunks (1,200 characters, 200 overlap), saves to `data/chunks.json`
2. **`embed_chunks.py`** — calls the OpenAI embeddings API (`text-embedding-3-small`) on every chunk, saves vectors to `data/embeddings.npy`
3. **`app.py`** — Flask web server; on each question it embeds the query, finds the top 5 chunks by cosine similarity, passes them to `gpt-4o-mini`, and returns the answer with sources

### Corpus

5 papers, 308 chunks:

| Paper | Chunks |
|---|---|
| Hopper et al., 2019.pdf | 78 |
| Kox et al., 2014.pdf | 49 |
| Nielsen et al., 2023.pdf | 50 |
| Russo et al., 2017.pdf | 65 |
| Vierra et al., 2022.pdf | 66 |

### Web UI features

- Answer cards with confidence badge (High / Medium / Low based on avg cosine similarity)
- Confidence tooltip explaining thresholds to non-technical users
- Low-confidence warning banner when max score < 0.40
- Keyword highlighting in retrieved chunks
- Source badges per answer
- Duplicate-source notice when multiple chunks come from the same paper
- Copy answer and export as Markdown buttons
- Show RAG details panel (retrieved chunks, scores, score bars)
- Latency and token count stats per answer
- Question history sidebar
- Corpus browser in sidebar — lists all indexed papers, click to open PDF
- Dark mode with localStorage persistence
- Print / PDF export

## Evaluation

Run `evaluate_semantic.py` to evaluate the semantic retrieval system and compare it against the keyword baseline:

```
python evaluate_semantic.py
```

Uses 5 hand-labelled questions in `eval_questions.json`, each with known correct chunk IDs. Reports Precision@K, Recall@K and MRR at K = 1, 3, 5, 10, side by side with the keyword baseline from `baseline_results.md`.

## Development history

Started with a keyword retrieval baseline evaluated against a small set of questions using Precision@K, Recall@K and MRR. Results are in `baseline_results.md`. Replaced by semantic embedding-based retrieval.

## Adding new papers to the corpus

1. Drop the PDF into the `documents/` folder
2. Run `python chunk_documents.py` — splits all PDFs into text chunks and saves to `data/chunks.json`
3. Run `python embed_chunks.py` — calls the OpenAI embeddings API and saves vectors to `data/embeddings.npy`
4. Restart the Flask app

The corpus list in the sidebar and all answers will automatically reflect the new papers. The chunk and embedding steps need to be re-run every time a paper is added or removed.

## Deployment

Configured for Railway (or Render). Requires `OPENAI_API_KEY` set as an environment variable in the platform dashboard.

```
railway login
railway init
railway up
```
