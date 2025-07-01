from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from nodes import agent, grade_documents, generate, rewrite, AgentState
from tools import tools

# Instantiate models once
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

chat_model = ChatGroq(model="llama-3.3-70b-versatile")
chat_model_with_tools = chat_model.bind_tools(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile()

result = {"messages": ["Test response"]} 