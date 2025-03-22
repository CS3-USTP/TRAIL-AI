from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch

# Define the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
embedding_model.save("models/all-mpnet-base-v2")

# Load and save CrossEncoder models
from sentence_transformers import CrossEncoder

relation_model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=device)
relation_model.save("models/nli-deberta-v3-base")

reranker_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1", device=device)
reranker_model.save("models/mxbai-rerank-xsmall-v1")
