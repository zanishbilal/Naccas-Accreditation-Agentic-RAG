from langchain_huggingface import HuggingFaceEmbeddings
import torch


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
        
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        model_kwargs={"trust_remote_code": True, "device": device},
        encode_kwargs={"normalize_embeddings": True},
    )