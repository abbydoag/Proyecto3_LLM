import json
from sentence_transformers import SentenceTransformer
import chromadb

#chunks
with open("data/processed/Stardew_data.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

#embeddings
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma_db")

try:
    client.delete_collection(name="stardew")
    print("Coleccion anterior borrada")
except Exception:
    pass

collection = client.create_collection(
    name="stardew",
    metadata={"hnsw:space": "cosine"}
)

texts = [c["text"] for c in chunks]
print(f"Total chunks: {len(texts)}")

#limite
BATCH_SIZE = 5000
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

for start in range(0, len(texts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(texts))
    print(f"Indexando batch {start}-{end} de {len(texts)}...")
    collection.add(
        ids=[f"chunk_{i}" for i in range(start, end)],
        embeddings=embeddings[start:end].tolist(),
        documents=texts[start:end],
        metadatas=[
            {"page": chunks[i]["page"], "section": chunks[i]["section"]}
            for i in range(start, end)
        ],
    )

print(f"\n{len(chunks)} chunks indexados en ChromaDB")