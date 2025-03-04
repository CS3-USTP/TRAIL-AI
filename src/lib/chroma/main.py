import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="USTP Handbook Semantic Search API")

# Load the embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Connect to ChromaDB
client = PersistentClient(path="src/lib/chroma/db")
collection = client.get_or_create_collection("ustp_handbook_2023")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 10
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}

@app.get("/query-metadata/{doc_id}")
def query_metadata(doc_id: str):
    """Retrieve a document by its metadata ID."""
    results = collection.get(ids=[doc_id])
    return results if results.get("documents") else {"error": "Document not found"}

@app.post("/semantic-search")
def semantic_search(request: QueryRequest):
    """Perform semantic search using a query."""
    query_embedding = model.encode(request.query, device=device, convert_to_numpy=True, normalize_embeddings=True)
    return collection.query(query_embeddings=[query_embedding.tolist()], n_results=request.n_results)
