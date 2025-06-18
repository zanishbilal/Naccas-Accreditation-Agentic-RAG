#Agentic RAG (Retrival agent_Langraph)


# Import required modules from LangChain and other libraries
import os
import getpass
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from IPython.display import Image, display
from langchain_core.messages import BaseMessage

pc = Pinecone(api_key="")
os.environ['PINECONE_API_KEY'] = ''
os.environ["GROQ_API_KEY"] = ""



# Connect to existing index and create retriever
index_name = "clara-qwen2-1p5b-index"
device = "cuda"
embedding_model = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    model_kwargs={"trust_remote_code": True, "device": device},
    encode_kwargs={"normalize_embeddings": True}
)
vectordb = PineconeVectorStore(index_name= index_name,embedding=embedding_model)
retriever= vectordb.as_retriever(search_kwargs={"k": 4})

# Following code for Agentic RAG structure

# Making a retriever tool that LLM will use if query is specific

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_policies",
    "Search and return accurate information about NACCAS policies.",
)


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


# Grade documents

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

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "ONLY respond by Formulating only 1 improved question, nothing else"
)

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