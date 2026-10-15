import json
from sentence_transformers import SentenceTransformer
import chromadb
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
if device == "cuda":
    model.half()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("egypt_legal")

BATCH_SIZE = 16 if device == "cuda" else 16
CHROMA_BATCH = 5000  # Chroma's per-call add() limit is 5461, stay under it

texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

for i in range(0, len(chunks), CHROMA_BATCH):
    batch_chunks = chunks[i:i + CHROMA_BATCH]
    batch_embeddings = embeddings[i:i + CHROMA_BATCH]

    collection.add(
        ids=[f"chunk_{j}" for j in range(i, i + len(batch_chunks))],
        embeddings=batch_embeddings.tolist(),
        documents=[c["text"] for c in batch_chunks],
        metadatas=[{"law_name": c["law_name"], "categories": c["categories"]} for c in batch_chunks]
    )
    print(f"Added {i + len(batch_chunks)} / {len(chunks)} to Chroma")

print(f"Indexed {len(chunks)} chunks. Vector store saved in ./chroma_db")
