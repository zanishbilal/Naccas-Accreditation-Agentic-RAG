# db_generator_recursive.py
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Set your Pinecone API key
os.environ['PINECONE_API_KEY'] = "your-pinecone-api-key"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# ---- Step 1: Load Documents ----
def load_documents(data_path):
    print(f"Loading PDFs from: {data_path}")
    return PyPDFDirectoryLoader(data_path).load()

pdfs_path = "./pdf"  # change this path as needed
pdfs = load_documents(pdfs_path)

# ---- Step 2: Split Text into Chunks ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600
)
splits = text_splitter.split_documents(documents=pdfs)
print(f"Number of chunks created: {len(splits)}")

# ---- Step 3: Initialize Embedding Model ----
device = "cuda"  # change to "cpu" if needed
embedding_model = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    model_kwargs={"trust_remote_code": True, "device": device},
    encode_kwargs={"normalize_embeddings": True}
)

# ---- Step 4: Store in Pinecone ----
index_name = "clara-qwen2"
if index_name not in pc.list_indexes():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Upload to Pinecone
vectordb = PineconeVectorStore.from_documents(
    documents=splits,
    index_name=index_name,
    embedding=embedding_model
)
print("Chunks uploaded to Pinecone successfully.")
