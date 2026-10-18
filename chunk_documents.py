from pathlib import Path
from pypdf import PdfReader
import json

DOCUMENTS_FOLDER = Path("documents")
DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

all_chunks = []

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

for pdf_path in DOCUMENTS_FOLDER.glob("*.pdf"):
    print(f"Reading {pdf_path.name}")

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        page_chunks = split_text(page_text)

        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
                "source": pdf_path.name,
                "page": page_number,
                "chunk_id": f"{pdf_path.stem}_p{page_number}_c{i}",
                "text": chunk
            })

output_path = DATA_FOLDER / "chunks.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(all_chunks)} chunks to {output_path}")