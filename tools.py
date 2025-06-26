"""
Tools for the NACCAS Policy Assistant including Pinecone retriever setup.
"""
import os
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME, RETRIEVER_K


# Set Pinecone API key
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"trust_remote_code": True, "device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Connect to index and build retriever
vectordb = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})

# LangChain retriever tool for LangGraph
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_policies",
    "Search and return accurate information about NACCAS policies.",
)