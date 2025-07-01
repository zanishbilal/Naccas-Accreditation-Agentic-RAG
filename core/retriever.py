# ✅ core/retriever.py
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_API_KEY, INDEX_NAME
from .embedding import get_embedding_model

embedding_model = get_embedding_model()
pc = Pinecone(api_key=PINECONE_API_KEY)
vectordb = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_policies",
    "Search and return accurate information about NACCAS policies."
)