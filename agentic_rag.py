"""#Agentic RAG (Retrival agent_Langraph)"""

# Install necessary libraries
# uncomment the following lines if you want to run on colab

# !pip install faiss-cpu
# !pip install -U bitsandbytes
# !pip install -U transformers accelerate
# !pip install -U langchain-huggingface
# # === INSTALLS ===
# !pip install -q langchain langchain-community pypdf transformers accelerate
# !pip install -qU langchain-groq
# !pip install streamlit
# !pip install -U --quiet langgraph langchain-text-splitters

# from google.colab import drive
# drive.mount('/content/drive')

# Import required modules from LangChain and other libraries
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from google import genai
from google.genai import types
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List
import os
import getpass

# Import the PyPDF loader from langchain_community
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Define a function to load all PDF documents from a directory
def load_documents(data_path):
    return PyPDFDirectoryLoader(data_path).load()

"""change path for pdf accordingly"""

# Define the path to the directory containing the PDF files and loading the PDF documents
# pdfs_path="/content/drive/MyDrive/Carla RAG/pdf"
pdfs_path="/content/drive/MyDrive/Clara Project/Clara docs"
pdfs=load_documents(pdfs_path)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Initialize=ing the text splitter to create overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# Splitting the loaded PDF documents into manageable text chunks
splits = text_splitter.split_documents(documents=pdfs)

# Loading a pre-trained sentence embedding model from HuggingFace
embedding_model = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# Creating a FAISS based vector store from the text chunks using the embedding model
vectordb = FAISS.from_documents(splits, embedding_model) #Faiss is indexing the vectors created from wordembeddings, also note that embeddings are also created from splits in this step

# Save the FAISS index to Google Drive (or any directory)
vectordb.save_local("/content/drive/MyDrive/Clara Project/Clara docs/QwenEmbeddings")
# save to drive and just retrieve if embeder is not changing

from langchain.vectorstores import FAISS

# Safely load your trusted FAISS index from local/Drive
vectordb = FAISS.load_local("/content/drive/MyDrive/Clara Project/Clara docs/QwenEmbeddings", embeddings=embedding_model,allow_dangerous_deserialization=True)
# Here deserialization means to convert a stored object (e.g in pickle format) to python code, which can be dangerous if file is from untrusted source

retriever=vectordb.as_retriever(search_kwargs={"k": 3})

"""Following code for Agentic RAG structure"""

# Making a retriever tool that LLM will use if query is specific
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_policies",
    "Search and return accurate information about NACCAS policies.",
)
# test tool
# retriever_tool.invoke({"query": "Is GED acceptable for admission?"})

"""Building the node and edge structure

"""

# Add groq key to use LLM
os.environ["GROQ_API_KEY"] = "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"

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

# LLM assigning
response_model = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5, max_tokens=512, timeout=None)

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}


"""Grade documents"""

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model("groq:meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5, max_tokens=512, timeout=None)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

REWRITE_PROMPT =(
    " You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval."
     "Look at the initial and formulate an improved question."
     "Here is the initial question: \n\n"
     "{question}"
     "Improved question with no preamble:"
)
# explicitly stated one improved question

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}



GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks regarding NACCAS policies. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use 25 sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

from langchain_core.messages import BaseMessage
import time

start_time = time.time()
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "A full-time student in a 1500-hour esthetics program (academic year of 900 clock hours, 30 weeks) fails to meet the minimum 70% cumulative grade average at the second evaluation period (900 clock hours, 30 weeks). The institution allows appeals for SAP determinations. What steps must the institution take, and what must its SAP policy include regarding the appeal process?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        # Check if 'messages' key exists and the last item is a BaseMessage before pretty printing
        if "messages" in update and update["messages"] and isinstance(update["messages"][-1], BaseMessage):
            update["messages"][-1].pretty_print()
        else:
            # Otherwise, print the update dictionary
            print(update)
        print("\n\n")
time.sleep(2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")