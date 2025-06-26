"""
Workflow definition for the NACCAS Policy Assistant using LangGraph.
"""
import os
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from tools import retriever_tool
from config import GROQ_API_KEY, RESPONSE_MODEL_NAME, MODEL_TEMPERATURE, MAX_TOKENS


# Set Groq API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize response model
response_model = init_chat_model(
    RESPONSE_MODEL_NAME, 
    temperature=MODEL_TEMPERATURE, 
    max_tokens=MAX_TOKENS
)

# Prompts
GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n ------- \n{question}\n ------- \n"
    "ONLY respond by Formulating only 1 improved question, nothing else"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks regarding NACCAS policies. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use 25 sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


class GradeDocuments(BaseModel):
    """Schema for document relevance grading."""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")


def generate_query_or_respond(state: MessagesState):
    """Generate a query or respond with tools."""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Grade the relevance of retrieved documents."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = response_model.with_structured_output(GradeDocuments).invoke([
        {"role": "user", "content": prompt}
    ])
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the question for better retrieval."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState):
    """Generate the final answer based on context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# Create workflow
workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges("generate_query_or_respond", tools_condition, {
    "tools": "retrieve",
    END: END,
})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile the graph
graph = workflow.compile()