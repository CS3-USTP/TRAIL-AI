from haystack.nodes import FARMReader

# Set Hugging Face token if needed
import os
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Load RoBERTa QA model
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

# Example text (document) and query
document = {
    "content": "The capital of France is Paris."
}
query = "What is the capital of France?"

# Get answers from the document

results = reader.predict(query=query, documents=document)

print(results['answers'])
