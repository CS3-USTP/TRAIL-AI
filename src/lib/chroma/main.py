import json
import torch
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor  # ✅ Offload blocking tasks

# ---------------------------- Initialize FastAPI ---------------------------- #

app = FastAPI(title="USTP Handbook Semantic Search API")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ Load the models ----------------------------- #

embedding_model = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    trust_remote_code=True,
    device=device   
)

coherence_model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-large",
    trust_remote_code=True,
    device=device
)

# --------------- ThreadPoolExecutor to offload blocking tasks --------------- #

executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------- Connect to ChromaDB --------------------------- #

client = PersistentClient(path="src/lib/chroma/db")
collection = client.get_collection("ustp_handbook_2023")


# ------------------------- Define the request models ------------------------ #

class QueryRequest(BaseModel):
    query: str


class CoherenceRequest(BaseModel):
    premise: str
    hypothesis: str


class SearchResult(BaseModel):
    document: str
    reference: str
    distance: float
    score: float


# --------------------------- Define the API routes -------------------------- #

@app.get("/")
async def read_root() -> Dict[str, str]:  # ✅ Now async
    """Welcome message."""
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}


@app.post("/coherence-check")
async def coherence_check(request: CoherenceRequest) -> Dict[str, Any]:
    """Check the coherence of the query with the provided context."""
    
    premise = request.premise
    hypothesis = request.hypothesis

    # Run model prediction asynchronously in the executor
    results = await asyncio.get_running_loop().run_in_executor(
        executor, 
        lambda: coherence_model.predict([(premise, hypothesis)])
    )
    # model can bulk queries, we take the first query result
    scores = results[0].tolist()
    
    # # get the highest score 
    # score = max(scores)
        
    # # get the highest score index in the scores list
    # max_index = scores.index(score)
    
    # # get the category of the highest score index
    # categories = ["contradiction", "neutral", "entailment"]
    # category = categories[max_index]    
     
    return JSONResponse(
        content={
            "results": [
                {"category": "contradiction", "score": scores[0]},
                {"category": "neutral", "score": scores[1]},
                {"category": "entailment", "score": scores[2]}
            ]
        },
        status_code=200
    )
    

@app.get("/query-metadata/{doc_id}")
async def query_metadata(doc_id: str) -> Dict[str, Any]:  # ✅ Now async
    """Retrieve a document by its metadata ID."""
    
    results = await asyncio.get_running_loop().run_in_executor(
        executor, collection.get, [doc_id]
    )  # ✅ Offloaded to thread pool
    
    if not results.get("documents"):
        raise HTTPException(status_code=404, detail="Document not found")

    return results


@app.post("/semantic-search")
async def semantic_search(request: QueryRequest) -> Dict[str, Any]:  # ✅ Now async
    """Perform semantic search and return only relevant results (distance < threshold)."""

    n_results = 10
    threshold = 1.35
    document = ""
    reference = ""
    distance = ""

    # ✅ Offload embedding generation to avoid blocking
    query_embedding = await asyncio.get_running_loop().run_in_executor(
        executor,
        lambda: model.encode(request.query, prompt_name="s2p_query", convert_to_numpy=True, normalize_embeddings=True)
    )

    # ✅ Offload ChromaDB query to avoid blocking
    response = await asyncio.get_running_loop().run_in_executor(
        executor,
        lambda: collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    )

    for chunk in zip(response["documents"][0], response["distances"][0], response["ids"][0]):
        
        print(chunk, "\n\n=====\n\n")
        
        if chunk[1] >= threshold:
            continue

        document  += chunk[0]      + "\n\n=====\n\n"
        distance  += str(chunk[1]) + ", "
        reference += chunk[2]      + ", "

    document  = document.strip()
    distance  = distance[:-2]
    reference = reference[:-2]

    if not document:
        return JSONResponse(content={"success": False, "message": "No relevant results found."}, status_code=200)

    return JSONResponse(content={"success": True, "document": document, "reference": reference, "distance": distance}, status_code=200)
