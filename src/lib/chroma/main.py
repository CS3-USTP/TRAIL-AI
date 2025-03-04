import json
import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI
app = FastAPI(title="USTP Handbook Semantic Search API")

# Load the embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Connect to ChromaDB
client = PersistentClient(path="src/lib/chroma/db")
collection = client.get_collection("ustp_handbook_2023")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 10
    
@app.get("/")
def read_root() -> Dict[str, str]:
    """Welcome message."""
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}

@app.get("/query-metadata/{doc_id}")
def query_metadata(doc_id: str) -> Dict[str, Any]:
    """Retrieve a document by its metadata ID."""
    results = collection.get(ids=[doc_id])
    return results if results.get("documents") else {"error": "Document not found"}

@app.post("/semantic-search")
def semantic_search(request: QueryRequest) -> Dict[str, Any]:
    """Perform semantic search and return only relevant results (distance < 1)."""
    
    # Generate the query embedding
    query_embedding = model.encode(
        request.query, 
        device=device, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    # Perform search in ChromaDB
    response = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=request.n_results
    )

    threshold = 1
    document = ""
    reference = ""
    distance = ""
    
    for chunk in zip(
        response["documents"][0], 
        response["distances"][0],
        response["ids"][0], 
        ):
        
        # Filter out results with distance greater than threshold
        if chunk[1] >= threshold: 
            continue

        # Concatenate the chunks 
        document  += chunk[0]      + "\n\n=====\n\n"
        distance  += str(chunk[1]) + ", "
        reference += chunk[2]      + ", "
        
    # Remove trailing comma and whitespace
    document  = document.strip()
    distance  = distance[:-2]
    reference = reference[:-2]
    
    if not document:
        return {"error": "No relevant results found"}
    return {
            "document":  document,
            "reference": reference,
            "distance":  distance
        }