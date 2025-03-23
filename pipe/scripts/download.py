from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch

# Define the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load and save SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
embedding_model.save("models/all-mpnet-base-v2")
del embedding_model

# Load and save CrossEncoder models
from sentence_transformers import CrossEncoder

relation_model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device=device)
relation_model.save("models/nli-deberta-v3-base")
del relation_model


reranker_model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1", device=device)
reranker_model.save("models/mxbai-rerank-base-v1")
del reranker_model