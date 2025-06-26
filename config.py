"""
Configuration file for API keys and environment variables.
Create a .env file in the project root and add your API keys there.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

# Pinecone Configuration
PINECONE_INDEX_NAME = "clara-qwen2"

# Model Configuration
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
RESPONSE_MODEL_NAME = "groq:meta-llama/llama-4-maverick-17b-128e-instruct"

# Model Parameters
MODEL_TEMPERATURE = 0.3
MAX_TOKENS = 512
RETRIEVER_K = 3