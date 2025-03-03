from fastapi import FastAPI, HTTPException, Query
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import json

# Initialize FastAPI app
app = FastAPI()

# Initialize GPU-supported embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Connect to ChromaDB
client = PersistentClient(path="db")  # Store data persistently
collection_name = "ustp_handbook_2023"
collection = client.get_or_create_collection(collection_name)


@app.get("/query")
def query_database(q: str = Query(..., description="Query string for semantic search")):
    try:
        query_embedding = model.encode(q, device=device, convert_to_numpy=True, normalize_embeddings=True)
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    try:
        results = collection.get(where={"id": doc_id})
        if not results["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
