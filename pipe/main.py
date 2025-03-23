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
import time
from enum import IntEnum
from datetime import datetime

# ---------------------------- Debug Configuration --------------------------- #

class DebugLevel(IntEnum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

# Set the debug level here (can be changed at runtime)
DEBUG_LEVEL = DebugLevel.INFO

def debug_print(level: DebugLevel, message: str, duration: Optional[float] = None):
    """Print debug messages with time, level and optional duration."""
    if level <= DEBUG_LEVEL:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_str = f"[{level.name}]".ljust(9)
        duration_str = f" ({duration:.4f}s)" if duration is not None else ""
        print(f"{timestamp} {level_str} {message}{duration_str}")

# ---------------------------- Initialize FastAPI ---------------------------- #

app = FastAPI(title="USTP Handbook Semantic Search API")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
debug_print(DebugLevel.INFO, f"Using device: {device}")


# ------------------------------ Load the models ----------------------------- #

debug_print(DebugLevel.INFO, "Loading models...")
start_time = time.time()

debug_print(DebugLevel.DEBUG, "Loading embedding model...")
embedding_model_start = time.time()
embedding_model = SentenceTransformer(
    "models/all-mpnet-base-v2",
    device=device   
)
embedding_model_duration = time.time() - embedding_model_start
debug_print(DebugLevel.INFO, "Embedding model loaded", embedding_model_duration)

debug_print(DebugLevel.DEBUG, "Loading relation model...")
relation_model_start = time.time()
relation_model = CrossEncoder(
    "models/nli-deberta-v3-base",
    device=device
)
relation_model_duration = time.time() - relation_model_start
debug_print(DebugLevel.INFO, "Relation model loaded", relation_model_duration)

debug_print(DebugLevel.DEBUG, "Loading reranker model...")
reranker_model_start = time.time()
reranker_model = CrossEncoder(
    'models/mxbai-rerank-base-v1',
    trust_remote_code=True,
    device=device
)
reranker_model_duration = time.time() - reranker_model_start
debug_print(DebugLevel.INFO, "Reranker model loaded", reranker_model_duration)

debug_print(DebugLevel.DEBUG, "Loading coherence model...")
coherence_model_start = time.time()
coherence_model = load("models/coherence.joblib")
coherence_model_duration = time.time() - coherence_model_start
debug_print(DebugLevel.INFO, "Coherence model loaded", coherence_model_duration)

total_model_loading_duration = time.time() - start_time
debug_print(DebugLevel.INFO, "All models loaded", total_model_loading_duration)


# ---------------------------- Connect to ChromaDB --------------------------- #

debug_print(DebugLevel.INFO, "Connecting to ChromaDB...")
db_connection_start = time.time()
client = PersistentClient(path="db/chroma")
collection = client.get_collection("ustp_handbook_2023")
db_connection_duration = time.time() - db_connection_start
debug_print(DebugLevel.INFO, f"Connected to ChromaDB collection: ustp_handbook_2023", db_connection_duration)


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
    debug_print(DebugLevel.DEBUG, f"Plotting 1D array with threshold={threshold}, mean={mean}")
    
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
    
    debug_print(DebugLevel.DEBUG, "Plot generation complete")


# Helper function to run CPU-bound tasks in a thread pool
async def run_in_threadpool(func, *args, **kwargs):
    """Run a CPU-bound function in a threadpool to avoid blocking the event loop."""
    debug_print(DebugLevel.TRACE, f"Running {func.__name__} in threadpool")
    start_time = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    duration = time.time() - start_time
    debug_print(DebugLevel.TRACE, f"Threadpool execution of {func.__name__} complete", duration)
    return result


# --------------------------- Define the API routes -------------------------- #

@app.get("/")
async def read_root() -> Dict[str, str]:
    """Welcome message."""
    debug_print(DebugLevel.INFO, "Root endpoint called")
    return {"message": "Welcome to the USTP Handbook Semantic Search API!"}


@app.post("/coherence-check")
async def coherence_check(request: CoherenceRequest) -> Dict[str, Any]:
    """Check the coherence of the query with the provided context."""
    debug_print(DebugLevel.INFO, f"Coherence check called with: premise={request.premise[:50]}..., hypothesis={request.hypothesis[:50]}...")
    
    premise = request.premise
    hypothesis = request.hypothesis

    # Run model prediction in a thread pool
    debug_print(DebugLevel.DEBUG, "Running relation model prediction")
    relation_start = time.time()
    results = await run_in_threadpool(relation_model.predict, [(premise, hypothesis)])
    relation_duration = time.time() - relation_start
    debug_print(DebugLevel.DEBUG, "Relation model prediction complete", relation_duration)
    
    # model can bulk queries, we take the first query result
    values = results[0].tolist()
    debug_print(DebugLevel.DEBUG, f"Relation values: contradiction={values[0]:.4f}, neutral={values[1]:.4f}, entailment={values[2]:.4f}")
    
    debug_print(DebugLevel.DEBUG, "Running coherence model prediction")
    coherence_start = time.time()
    coherence = await run_in_threadpool(predict, coherence_model, [values])
    coherence_duration = time.time() - coherence_start
    debug_print(DebugLevel.DEBUG, f"Coherence prediction: {coherence}", coherence_duration)
    
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
    debug_print(DebugLevel.INFO, f"Metadata query for document ID: {doc_id}")
    
    # ChromaDB operations in a thread pool
    debug_print(DebugLevel.DEBUG, "Fetching document from ChromaDB")
    chroma_start = time.time()
    results = await run_in_threadpool(collection.get, [doc_id])
    chroma_duration = time.time() - chroma_start
    debug_print(DebugLevel.DEBUG, f"ChromaDB fetch complete", chroma_duration)
    
    if not results.get("documents"):
        debug_print(DebugLevel.WARNING, f"Document not found: {doc_id}")
        raise HTTPException(status_code=404, detail="Document not found")

    debug_print(DebugLevel.INFO, f"Document found, returning metadata")
    return results


@app.post("/semantic-search")
async def semantic_search(request: QueryRequest) -> JSONResponse:
    """Perform semantic search and return only relevant results."""
    debug_print(DebugLevel.INFO, f"Semantic search called with query: {request.query}")
    
    try:
        query = request.query
        debug_print(DebugLevel.INFO, f"Creating semantic search pipeline for query: {query}")
        pipeline = SemanticSearchPipeline(query)
        
        search_start = time.time()
        search_results = await pipeline.execute()
        search_duration = time.time() - search_start
        
        if search_results.get("success", False):
            num_results = len(search_results.get("reference", "").split(",")) if search_results.get("reference") else 0
            debug_print(DebugLevel.INFO, f"Search complete with {num_results} results", search_duration)
        else:
            debug_print(DebugLevel.WARNING, f"Search unsuccessful: {search_results.get('message', 'Unknown error')}", search_duration)
            
        return JSONResponse(
            content=search_results,
            status_code=200
        )
    except Exception as e:
        debug_print(DebugLevel.ERROR, f"Error in semantic search: {str(e)}")
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
        debug_print(DebugLevel.DEBUG, f"Initialized SemanticSearchPipeline with query: {query}")
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the complete search pipeline."""
        debug_print(DebugLevel.INFO, f"Executing search pipeline for query: {self.query}")
        
        debug_print(DebugLevel.DEBUG, "Starting document retrieval stage")
        retrieval_start = time.time()
        await self.retrieve_documents()
        retrieval_duration = time.time() - retrieval_start
        debug_print(DebugLevel.DEBUG, "Document retrieval complete", retrieval_duration)
        
        if not self.results.get("documents"):
            debug_print(DebugLevel.WARNING, "No documents found in retrieval stage")
            return {"success": False, "message": "No relevant results found."}
        
        num_retrieved = len(self.results.get("documents", []))
        debug_print(DebugLevel.INFO, f"Retrieved {num_retrieved} documents, starting reranking stage")
        
        reranking_start = time.time()
        await self.rerank_documents()
        reranking_duration = time.time() - reranking_start
        debug_print(DebugLevel.DEBUG, "Document reranking complete", reranking_duration)
        
        debug_print(DebugLevel.DEBUG, "Formatting results")
        format_result = self.format_results()
        debug_print(DebugLevel.DEBUG, "Results formatting complete")
        
        return format_result
    
    async def retrieve_documents(self) -> None:
        """Retrieve relevant documents based on embedding similarity."""
        debug_print(DebugLevel.DEBUG, f"Generating embedding for query: {self.query}")
        
        # Generate query embedding
        encoding_start = time.time()
        query_embedding = await run_in_threadpool(
            embedding_model.encode,
            self.query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        encoding_duration = time.time() - encoding_start
        debug_print(DebugLevel.DEBUG, f"Query embedding complete, shape: {query_embedding.shape}", encoding_duration)
        
        # Query ChromaDB
        debug_print(DebugLevel.DEBUG, f"Querying ChromaDB with n_results={self.INITIAL_RESULTS_COUNT}")
        chroma_start = time.time()
        response = await run_in_threadpool(
            collection.query,
            query_embeddings=[query_embedding.tolist()],
            n_results=self.INITIAL_RESULTS_COUNT
        )
        chroma_duration = time.time() - chroma_start
        debug_print(DebugLevel.DEBUG, "ChromaDB query complete", chroma_duration)
        
        # Extract results
        documents = response.get("documents", [[]])[0]
        references = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        
        debug_print(DebugLevel.DEBUG, f"Retrieved {len(documents)} documents from ChromaDB")
        if documents:
            avg_distance = sum(distances) / len(distances)
            debug_print(DebugLevel.DEBUG, f"Distance stats: min={min(distances):.4f}, max={max(distances):.4f}, avg={avg_distance:.4f}")
        
        # Filter by threshold
        debug_print(DebugLevel.DEBUG, f"Filtering documents by distance threshold: {self.EMBEDDING_THRESHOLD}")
        filtered_results = self._filter_by_threshold(
            documents, references, distances, 
            threshold=self.EMBEDDING_THRESHOLD,
            lower_is_better=True
        )
        
        filtered_count = len(filtered_results.get("documents", []))
        debug_print(DebugLevel.INFO, f"After filtering: {filtered_count}/{len(documents)} documents remain")
        
        self.results = filtered_results
    
    async def rerank_documents(self) -> None:
        """Rerank retrieved documents using the reranker model."""
        documents = self.results.get("documents", [])
        references = self.results.get("references", [])
        distances = self.results.get("distances", [])
        
        if not documents:
            debug_print(DebugLevel.WARNING, "No documents to rerank")
            return
            
        debug_print(DebugLevel.INFO, f"Reranking {len(documents)} documents")
        ranked_results = []
        
        # Process in batches
        num_batches = (len(documents) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        debug_print(DebugLevel.DEBUG, f"Processing in {num_batches} batches of size {self.BATCH_SIZE}")
        
        for i in range(0, len(documents), self.BATCH_SIZE):
            batch_idx = i // self.BATCH_SIZE + 1
            batch_start = time.time()
            
            batch_docs = documents[i:i+self.BATCH_SIZE]
            batch_refs = references[i:i+self.BATCH_SIZE]
            batch_dists = distances[i:i+self.BATCH_SIZE]
            
            debug_print(DebugLevel.DEBUG, f"Processing batch {batch_idx}/{num_batches} with {len(batch_docs)} documents")
            
            # Prepare batch inputs for reranker
            batch_inputs = [(self.query, doc) for doc in batch_docs]
            
            # Get reranker scores
            reranker_start = time.time()
            batch_scores = await run_in_threadpool(reranker_model.predict, batch_inputs)
            reranker_duration = time.time() - reranker_start
            debug_print(DebugLevel.DEBUG, f"Reranking batch {batch_idx} complete", reranker_duration)
            
            # Collect results
            for doc, ref, dist, score in zip(batch_docs, batch_refs, batch_dists, batch_scores):
                ranked_results.append((doc, ref, dist, float(score)))
            
            batch_duration = time.time() - batch_start
            debug_print(DebugLevel.DEBUG, f"Batch {batch_idx}/{num_batches} processed", batch_duration)
        
        # Sort by reranker score (higher is better)
        debug_print(DebugLevel.DEBUG, "Sorting results by reranker score")
        ranked_results.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (doc, ref, dist, score) in enumerate(ranked_results, 1):
            debug_print(DebugLevel.TRACE, "\n====================\n")
            debug_print(DebugLevel.TRACE, f"Rank: {rank}") 
            debug_print(DebugLevel.TRACE, f"Reference: {ref}")
            debug_print(DebugLevel.TRACE, f"Distance: {dist}")
            debug_print(DebugLevel.TRACE, f"Score: {score}")
            debug_print(DebugLevel.TRACE, f"Document: {doc[:100]}...")
            debug_print(DebugLevel.TRACE, "\n====================\n")
        
        # Unpack sorted results
        if ranked_results:
            sorted_docs, sorted_refs, sorted_dists, sorted_scores = zip(*ranked_results)
            
            # Calculate stats for debug
            min_score = min(sorted_scores)
            max_score = max(sorted_scores)
            mean_score = sum(sorted_scores) / len(sorted_scores)
            debug_print(DebugLevel.DEBUG, f"Reranker scores: min={min_score:.4f}, max={max_score:.4f}, mean={mean_score:.4f}")
            
            # Optional: visualize scores
            debug_print(DebugLevel.DEBUG, "Visualizing reranker scores")
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
                debug_print(DebugLevel.DEBUG, f"Using mean-based threshold: {threshold:.4f}")
            else:
                threshold = self.RERANKER_THRESHOLD
                debug_print(DebugLevel.DEBUG, f"Using fixed threshold: {threshold:.4f}")
                
            debug_print(DebugLevel.DEBUG, "Filtering results by reranker score threshold")
            filtered_results = self._filter_by_threshold(
                filtered_results["documents"],
                filtered_results["references"],
                filtered_results["distances"],
                threshold=threshold,
                scores=filtered_results["scores"],
                lower_is_better=False
            )
            
            filtered_count = len(filtered_results.get("documents", []))
            debug_print(DebugLevel.INFO, f"After reranking filter: {filtered_count}/{len(sorted_scores)} documents remain")
            
            self.results = filtered_results
    
    def format_results(self) -> Dict[str, Any]:
        """Format the search results for API response."""
        documents = self.results.get("documents", [])
        references = self.results.get("references", [])
        distances = self.results.get("distances", [])
        scores = self.results.get("scores", [])
        
        if not documents:
            debug_print(DebugLevel.WARNING, "No documents to format in results")
            return {"success": False, "message": "No relevant results found."}
        
        debug_print(DebugLevel.INFO, f"Formatting {len(documents)} documents for response")
        
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
        debug_print(DebugLevel.DEBUG, f"Filtering results with threshold={threshold}, lower_is_better={lower_is_better}")
        
        start_time = time.time()
        filtered_docs = []
        filtered_refs = []
        filtered_dists = []
        filtered_scores = []
        
        # Determine which values to use for filtering
        filter_values = scores if scores is not None else distances
        
        # Apply appropriate comparison based on whether lower or higher is better
        for i, value in enumerate(filter_values):
            comparison_result = value < threshold if lower_is_better else value > threshold
            if comparison_result:
                filtered_docs.append(documents[i])
                filtered_refs.append(references[i])
                filtered_dists.append(distances[i])
                if scores:
                    filtered_scores.append(scores[i])
        
        duration = time.time() - start_time
        debug_print(DebugLevel.DEBUG, f"Filtered {len(filtered_docs)}/{len(documents)} results", duration)
        
        result = {
            "documents": filtered_docs,
            "references": filtered_refs,
            "distances": filtered_dists,
        }
        
        if scores:
            result["scores"] = filtered_scores
            
        return result


# Add debug middleware to log all incoming requests
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    method = request.method
    url = request.url.path
    
    debug_print(DebugLevel.INFO, f"Request: {method} {url}")
    
    # Extract and log request body for debugging
    try:
        body = await request.body()
        if body:
            body_text = body.decode()
            debug_print(DebugLevel.DEBUG, f"Request body: {body_text[:200]}...")
    except Exception as e:
        debug_print(DebugLevel.ERROR, f"Could not read request body: {str(e)}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    status_code = response.status_code
    debug_print(DebugLevel.INFO, f"Response: {status_code} for {method} {url}", duration)
    
    return response