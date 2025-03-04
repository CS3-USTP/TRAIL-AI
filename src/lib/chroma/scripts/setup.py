import csv
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Initialize GPU-supported embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Connect to ChromaDB
client = PersistentClient(path="../db")  # Store data persistently
collection_name = "ustp_handbook_2023"

# Delete collection if it exists
try:
    client.delete_collection(collection_name)
    print("Deleted existing collection.")
except:
    print("No existing collection to delete.")

# Create a new collection
print("Creating new collection...")
collection = client.get_or_create_collection(collection_name)
print("Collection created.")

# Read and process the CSV file
with open("data/chunks.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    
    for index, row in enumerate(reader):
        # Skip the header
        if index == 0:
            continue
        
        chunk = "\n".join(cell.strip() for cell in row).strip()
        if chunk:
            print(f"Processing chunk {index}...")
            
            # Generate embedding for the current chunk
            embedding = model.encode(chunk, device=device, convert_to_numpy=True, normalize_embeddings=True)
            
            # Add to ChromaDB
            collection.add(ids=[f"id-{index}"], documents=[chunk], embeddings=[embedding.tolist()])
            print(f"Chunk {index} added to ChromaDB.")

print("All documents processed.")
