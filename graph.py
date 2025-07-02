from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from nodes import agent, grade_document, generate, rewrite, AgentState
from tools import build_dynamic_retriever_tool

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
# For testing, hardcode a tool for a known URL
test_tool = build_dynamic_retriever_tool("https://scikit-learn.org/stable/modules/kernel_ridge.html")
retrieve = ToolNode([test_tool])
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
    grade_document,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")
graph = workflow.compile() 