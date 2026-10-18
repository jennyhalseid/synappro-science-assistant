# Keyword Retrieval Baseline

## Results

| K | Mean Precision | Mean Recall | MRR |
|---|---:|---:|---:|
| 1 | 0.80 | 0.43 | 0.80 |
| 3 | 0.33 | 0.50 | 0.80 |
| 5 | 0.24 | 0.57 | 0.80 |
| 10 | 0.16 | 0.70 | 0.80 |

## Interpretation

As K increases, recall improves because the system retrieves more chunks, making it more likely to include the correct one.

However, precision decreases because more irrelevant chunks are included.

MRR stays at 0.80, suggesting that when the keyword system finds a relevant chunk, it often ranks it near the top.

This keyword baseline gives us something to compare against later when we add semantic embeddings.