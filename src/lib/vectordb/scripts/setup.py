import csv
import torch
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Initialize GPU-supported embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load embedding model
model = SentenceTransformer(
    "NovaSearch/stella_en_400M_v5",
    trust_remote_code=True,
    device=device   
)

# Connect to ChromaDB
client = PersistentClient(path="../storage")  # Store data persistently
collection_name = "ustp_handbook_2023"

# Delete collection if it exists
try:
    client.delete_collection(collection_name)
    print("Deleted existing collection.")
except:
    print("No existing collection to delete.")

# Create a new collection
print("Creating new collection...")
collection = client.create_collection(collection_name)
print("Collection created.")

# Read CSV file and store chunks
chunks = []
with open("../data/chunks.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    
    for index, row in enumerate(reader):
        if index == 0:
            continue  # Skip header
        
        chunk = "\n".join(cell.strip() for cell in row).strip()
        if chunk:
            chunks.append((index, chunk))

# Process embeddings and store in ChromaDB
for index, chunk in chunks:
    print(f"Processing chunk {index}...")
    embedding = model.encode(chunk, prompt_name="s2p_query", convert_to_numpy=True, normalize_embeddings=True)
    collection.add(ids=[f"id-{index}"], documents=[chunk], embeddings=[embedding.tolist()])
    print(f"Chunk {index} added to ChromaDB.")

print("All documents processed.")
