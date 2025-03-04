import json
import csv
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Initialize GPU-supported embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading the model...")
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Connect to ChromaDB
print("Connecting to ChromaDB...")
client = PersistentClient(path="../db")  # Store data persistently
collection_name = "ustp_handbook_2023"

# Get the collection
print("Getting the collection...")
collection = client.get_or_create_collection(collection_name)

# Start input loop for testing
print("\nType your query and press Enter (or type 'exit' to quit):")
while True:
    query_text = input("\nQuery: ").strip()
    
    if query_text.lower() == "exit":
        print("Exiting program...")
        break

    print("Processing query...")

    # Encode the query
    query_embedding = model.encode(query_text, device=device, convert_to_numpy=True, normalize_embeddings=True)

    # Query ChromaDB
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10)

    # Print results
    print(json.dumps(results, indent=4, sort_keys=True))
