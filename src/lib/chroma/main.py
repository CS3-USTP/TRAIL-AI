import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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

@app.get("/")
def read_root() -> Dict[str, str]:
    """Welcome message."""
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}

@app.get("/query-metadata/{doc_id}")
def query_metadata(doc_id: str) -> Dict[str, Any]:
    """Retrieve a document by its metadata ID."""
    results = collection.get(ids=[doc_id])
    
    if not results.get("documents"):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return results

@app.post("/semantic-search")
def semantic_search(request: QueryRequest) -> Dict[str, Any]:
    """Perform semantic search and return only relevant results (distance < threshold)."""
    
    n_results = 20
    threshold = 0.94
    document = ""
    reference = ""
    distance = ""

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
        n_results=n_results
    )
    
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
        return JSONResponse(
            content={"success": False, "message": "No relevant results found."},
            status_code=200  # Keeping status as 200 to indicate a successful request
        )

    return JSONResponse(
        content={
            "success": True,
            "document": document,
            "reference": reference,
            "distance": distance
        },
        status_code=200
    )
