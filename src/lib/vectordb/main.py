from src.lib.coherence.main import predict
import plotext as plt
import numpy as np
import torch
from joblib import load
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
import asyncio

# ---------------------------- Initialize FastAPI ---------------------------- #

app = FastAPI(title="USTP Handbook Semantic Search API")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ Load the models ----------------------------- #

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    trust_remote_code=True,
    device=device   
)

relation_model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base",
    trust_remote_code=True,
    device=device
)

reranker_model = CrossEncoder(
    'mixedbread-ai/mxbai-rerank-large-v1',
    trust_remote_code=True,
    device=device
)

coherence_model = load("src/lib/coherence/out/model.joblib")


# ---------------------------- Connect to ChromaDB --------------------------- #

client = PersistentClient(path="src/lib/vectordb/storage")
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


def plot_1d_array_with_threshold(data: list[float], threshold: float) -> None:
    """
    Plots a smooth 1D array (or list) in the terminal using plotext, with an average line and manual legend.

    Parameters:
        data (list or numpy.ndarray): The 1D data to be plotted.
    """

    # Generate x-axis indices
    x = list(range(len(data)))

    # Clear previous plots
    plt.clear_data()

    # Enable dark mode
    plt.theme("dark")

    # Scatter plot for thick points with 'O' marker
    plt.scatter(x, data, marker="◆", color="blue")

    # Average line
    plt.plot(x, [threshold] * len(data), color="red", marker="┉	")

    # Customize the plot
    plt.title("1D Array Plot with Average Line")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(0, max(data) * 1)  # Adjust y-axis limits for better visibility
    plt.show()


# Helper function to run CPU-bound tasks in a thread pool
async def run_in_threadpool(func, *args, **kwargs):
    """Run a CPU-bound function in a threadpool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# --------------------------- Define the API routes -------------------------- #

@app.get("/")
async def read_root() -> Dict[str, str]:
    """Welcome message."""
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}


@app.post("/coherence-check")
async def coherence_check(request: CoherenceRequest) -> Dict[str, Any]:
    """Check the coherence of the query with the provided context."""
    
    premise = request.premise
    hypothesis = request.hypothesis

    # Run model prediction in a thread pool
    results = await run_in_threadpool(relation_model.predict, [(premise, hypothesis)])
    # model can bulk queries, we take the first query result
    values = results[0].tolist()
    
    coherence = await run_in_threadpool(predict, coherence_model, [values])

    print(f"Coherence: {coherence}")
    
    return JSONResponse(content={
        "coherence": coherence,
        "values": {
            "contradiction": values[0],
            "neutral":       values[1],
            "entailment":    values[2]
        }
        }, 
        status_code=200)


@app.get("/query-metadata/{doc_id}")
async def query_metadata(doc_id: str) -> Dict[str, Any]:
    """Retrieve a document by its metadata ID."""
    
    # ChromaDB operations in a thread pool
    results = await run_in_threadpool(collection.get, [doc_id])
    
    if not results.get("documents"):
        raise HTTPException(status_code=404, detail="Document not found")

    return results


@app.post("/semantic-search")
async def semantic_search(request: QueryRequest) -> JSONResponse:
    """Perform semantic search and return only relevant results."""

    query = request.query
    response = await query_paragraph(query)
    response = await rerank_response(query, response)
    response = await format_response(response)

    return response


async def query_paragraph(query: str) -> Dict[str, Any]:
    """Retrieve the document using the query embedding."""
    
    n_results = 20

    # Generate embeddings in a thread pool
    query_embedding = await run_in_threadpool(
        embedding_model.encode,
        query, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    # Query ChromaDB in a thread pool
    response = await run_in_threadpool(
        collection.query,
        query_embeddings=[query_embedding.tolist()], 
        n_results=n_results
    )
    
    return response


async def rerank_response(query: str, response: Dict[str, Any]) -> Dict[str, Any]:
    """Rerank the search results using a reranker model."""
    
    documents = response.get("documents", [[]])[0]
    references = response.get("ids", [[]])[0]
    distances = response.get("distances", [[]])[0]

    ranked_results = []

    # Compute scores - process in batches to avoid too many individual thread pool calls
    batch_size = 5
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_refs = references[i:i+batch_size]
        batch_dists = distances[i:i+batch_size]
        
        # Prepare batch inputs
        batch_inputs = [(query, doc) for doc in batch_docs]
        
        # Run prediction in thread pool
        batch_scores = await run_in_threadpool(reranker_model.predict, batch_inputs)
        
        # Add results to ranked_results
        for doc, ref, dist, score in zip(batch_docs, batch_refs, batch_dists, batch_scores):
            ranked_results.append((doc, ref, dist, float(score)))

    # Sort results based on the score (higher is better)
    ranked_results.sort(key=lambda x: x[3], reverse=True)

    # Extract sorted values
    sorted_documents, sorted_references, sorted_distances, sorted_scores = zip(*ranked_results)

    # Set the average score as the threshold
    threshold = np.mean(sorted_scores)

    # Debugging print (optional)
    for i, (doc, ref, dist, score) in enumerate(ranked_results):
        print("\n====\n")
        print(f"Rank: {i}")
        print(f"Distance: {dist}")
        print(f"Score: {score}")
        print(f"Document:\n{doc}")
        print("\n====\n")
    
    # Run plot in thread pool to avoid blocking
    await run_in_threadpool(plot_1d_array_with_threshold, sorted_scores, threshold)
    
    # filter out results below the threshold
    sorted_documents  = [doc   for doc,  score  in zip(sorted_documents,  sorted_scores) if score > threshold]
    sorted_references = [ref   for ref,  score  in zip(sorted_references, sorted_scores) if score > threshold]
    sorted_distances  = [dist  for dist, score  in zip(sorted_distances,  sorted_scores) if score > threshold]
    sorted_scores     = [score for score        in                        sorted_scores  if score > threshold]

    return {
        "documents": sorted_documents,
        "ids":       sorted_references,
        "distances": sorted_distances,
        "scores":    sorted_scores
    }


async def format_response(response: Dict[str, Any]) -> JSONResponse:
    """Format the response to be returned to the client."""
    
    documents  = response.get("documents") # list of documents
    distances  = response.get("distances") # list of distances
    references = response.get("ids")       # list of references
    scores     = response.get("scores")    # list of scores
    
    if not documents:
        return JSONResponse(
            content={"success": False, "message": "No relevant results found."},
            status_code=200
        )
    
    return JSONResponse(
        content={
            "success": True,
            "document": "\n\n".join(map(str, documents)),
            "reference": ", ".join(map(str, references)),
            "distance": ", ".join(map(str, distances)),
            "score": ", ".join(map(str, scores))    
        },
        status_code=200
    )