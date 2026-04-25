import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

# Load your .env file (ensure OPENAI_API_KEY is inside)
load_dotenv()

# 1. Initialize the MCP Server
# This is what Boomi will connect to
mcp = FastMCP("TherapyChatbotServer")

# 2. Setup your LLM
llm = init_chat_model("openai:gpt-4o-mini")

# --- YOUR GRAPH LOGIC ---

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ..., 
        description="Classify if message is emotional or logical"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    
    # Handle both string and object formats for safety
    content = last_message["content"] if isinstance(last_message, dict) else last_message.content
    
    result = classifier_llm.invoke([
        {"role": "system", "content": "Classify the user message as 'emotional' or 'logical'."},
        {"role": "user", "content": content}
    ])
    return {"message_type": result.message_type}

def router(state: State):
    return state.get("message_type", "logical")

def therapist_agent(state: State):
    reply = llm.invoke([
        {"role": "system", "content": "You are a compassionate therapist. Show empathy."},
        state["messages"][-1]
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    reply = llm.invoke([
        {"role": "system", "content": "You are a purely logical assistant. Focus on facts."},
        state["messages"][-1]
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

# --- ASSEMBLE GRAPH ---

builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("therapist", therapist_agent)
builder.add_node("logical", logical_agent)

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier", 
    router, 
    {"emotional": "therapist", "logical": "logical"}
)
builder.add_edge("therapist", END)
builder.add_edge("logical", END)

graph = builder.compile()

# --- THE MCP TOOL ---
# This exposes your graph to Boomi/Claude

@mcp.tool()
def process_message(user_input: str) -> str:
    """
    Categorizes a user message and provides either a compassionate 
    emotional response or a factual logical response.
    """
    # Create the initial state for the graph
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    
    # Run the graph
    result = graph.invoke(inputs)
    
    # Return the final assistant message
    return result["messages"][-1].content

if __name__ == "__main__":
    # Run as HTTP server for Boomi access
    # Mac/Boomi URL will be: http://localhost:8000/mcp
    mcp.run(transport="http", host="0.0.0.0", port=8000, stateless_http=True)