from lib.coherence.utils import predict
import plotext as plt
import numpy as np
import torch
from joblib import load
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio

# ---------------------------- Initialize FastAPI ---------------------------- #

app = FastAPI(title="USTP Handbook Semantic Search API")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ Load the models ----------------------------- #

embedding_model = SentenceTransformer(
    "models/all-mpnet-base-v2",
    device=device   
)

relation_model = CrossEncoder(
    "models/nli-deberta-v3-base",
    device=device
)

reranker_model = CrossEncoder(
    'models/mxbai-rerank-xsmall-v1',
    trust_remote_code=True,
    device=device
)

coherence_model = load("models/coherence.joblib")


# ---------------------------- Connect to ChromaDB --------------------------- #

client = PersistentClient(path="db/chroma")
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


def plot_1d_array_with_threshold(data: list[float], threshold: float, mean: float) -> None:
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
    plt.plot(x, [mean] * len(data), color="orange", marker="┉")

    # Threshold line
    plt.plot(x, [threshold] * len(data), color="red", marker="┉")


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
    try:
        query = request.query
        search_results = await SemanticSearchPipeline(query).execute()
        return JSONResponse(
            content=search_results,
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"Error: {str(e)}"},
            status_code=500
        )


class SemanticSearchPipeline:
    """Pipeline for semantic search with embedding, reranking, and formatting stages."""
    
    # Configuration constants
    INITIAL_RESULTS_COUNT = 10
    EMBEDDING_THRESHOLD = 1.5
    RERANKER_THRESHOLD = 0.005
    BATCH_SIZE = 5
    MAX_RESULTS_WITHOUT_MEAN_FILTER = 5
    
    def __init__(self, query: str):
        self.query = query
        self.results = {}
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the complete search pipeline."""
        await self.retrieve_documents()
        
        if not self.results.get("documents"):
            return {"success": False, "message": "No relevant results found."}
            
        await self.rerank_documents()
        return self.format_results()
    
    async def retrieve_documents(self) -> None:
        """Retrieve relevant documents based on embedding similarity."""
        # Generate query embedding
        query_embedding = await run_in_threadpool(
            embedding_model.encode,
            self.query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Query ChromaDB
        response = await run_in_threadpool(
            collection.query,
            query_embeddings=[query_embedding.tolist()],
            n_results=self.INITIAL_RESULTS_COUNT
        )
        
        # Extract results
        documents = response.get("documents", [[]])[0]
        references = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        
        # Filter by threshold
        filtered_results = self._filter_by_threshold(
            documents, references, distances, 
            threshold=self.EMBEDDING_THRESHOLD,
            lower_is_better=True
        )
        
        self.results = filtered_results
    
    async def rerank_documents(self) -> None:
        """Rerank retrieved documents using the reranker model."""
        documents = self.results.get("documents", [])
        references = self.results.get("references", [])
        distances = self.results.get("distances", [])
        
        if not documents:
            return
            
        ranked_results = []
        
        # Process in batches
        for i in range(0, len(documents), self.BATCH_SIZE):
            batch_docs = documents[i:i+self.BATCH_SIZE]
            batch_refs = references[i:i+self.BATCH_SIZE]
            batch_dists = distances[i:i+self.BATCH_SIZE]
            
            # Prepare batch inputs for reranker
            batch_inputs = [(self.query, doc) for doc in batch_docs]
            
            # Get reranker scores
            batch_scores = await run_in_threadpool(reranker_model.predict, batch_inputs)
            
            # Collect results
            for doc, ref, dist, score in zip(batch_docs, batch_refs, batch_dists, batch_scores):
                ranked_results.append((doc, ref, dist, float(score)))
        
        # Sort by reranker score (higher is better)
        ranked_results.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (doc, ref, dist, score) in enumerate(ranked_results, 1):
            print("\n====================\n")
            print(f"Rank: {rank}") 
            print(f"Reference: {ref}")
            print(f"Distance: {dist}")
            print(f"Score: {score}")
            print(f"Document: {doc}")
            print("\n====================\n")
        
        
        # Unpack sorted results
        if ranked_results:
            sorted_docs, sorted_refs, sorted_dists, sorted_scores = zip(*ranked_results)
            
            # Optional: visualize scores
            mean_score = sum(sorted_scores) / len(sorted_scores)
            await run_in_threadpool(
                plot_1d_array_with_threshold, 
                sorted_scores, self.RERANKER_THRESHOLD, mean_score
            )
            
            filtered_results = {
                "documents": list(sorted_docs),
                "references": list(sorted_refs),
                "distances": list(sorted_dists),
                "scores": list(sorted_scores)
            }
            
            # Determine the appropriate threshold to use
            # If more than MAX_RESULTS_WITHOUT_MEAN_FILTER results, use mean as threshold
            if len(sorted_scores) > self.MAX_RESULTS_WITHOUT_MEAN_FILTER:
                threshold = max(mean_score, self.RERANKER_THRESHOLD)
            else:
                threshold = self.RERANKER_THRESHOLD
                
            filtered_results = self._filter_by_threshold(
                filtered_results["documents"],
                filtered_results["references"],
                filtered_results["distances"],
                threshold=threshold,
                scores=filtered_results["scores"],
                lower_is_better=False
            )
            
            self.results = filtered_results
    
    def format_results(self) -> Dict[str, Any]:
        """Format the search results for API response."""
        documents = self.results.get("documents", [])
        references = self.results.get("references", [])
        distances = self.results.get("distances", [])
        scores = self.results.get("scores", [])
        
        if not documents:
            return {"success": False, "message": "No relevant results found."}
        
        return {
            "success": True,
            "document": "\n\n".join(map(str, documents)),
            "reference": ", ".join(map(str, references)),
            "distance": ", ".join(map(str, distances)),
            "score": ", ".join(map(str, scores))
        }
    
    @staticmethod
    def _filter_by_threshold(
        documents: List[str],
        references: List[str],
        distances: List[float],
        threshold: float,
        scores: Optional[List[float]] = None,
        lower_is_better: bool = True
    ) -> Dict[str, List]:
        """Filter results based on a threshold value.
        
        Args:
            documents: List of document texts
            references: List of document IDs
            distances: List of distance scores
            threshold: Cutoff threshold
            scores: Optional secondary scores (e.g., from reranker)
            lower_is_better: If True, keep values below threshold (for distances)
                            If False, keep values above threshold (for reranker scores)
        
        Returns:
            Filtered results as a dictionary
        """
        filtered_docs = []
        filtered_refs = []
        filtered_dists = []
        filtered_scores = []
        
        # Determine which values to use for filtering
        filter_values = scores if scores is not None else distances
        
        # Apply appropriate comparison based on whether lower or higher is better
        for i, value in enumerate(filter_values):
            if (lower_is_better and value < threshold) or (not lower_is_better and value > threshold):
                filtered_docs.append(documents[i])
                filtered_refs.append(references[i])
                filtered_dists.append(distances[i])
                if scores:
                    filtered_scores.append(scores[i])
        
        result = {
            "documents": filtered_docs,
            "references": filtered_refs,
            "distances": filtered_dists,
        }
        
        if scores:
            result["scores"] = filtered_scores
            
        return result