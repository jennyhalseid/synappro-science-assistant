import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

DATA_FOLDER = Path("data")
CHUNKS_PATH = DATA_FOLDER / "chunks.json"
EMBEDDINGS_PATH = DATA_FOLDER / "embeddings.npy"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

print(f"Creating embeddings for {len(texts)} chunks...")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = np.array([item.embedding for item in response.data])

np.save(EMBEDDINGS_PATH, embeddings)

print(f"Saved embeddings to {EMBEDDINGS_PATH}")
print(f"Embedding shape: {embeddings.shape}")