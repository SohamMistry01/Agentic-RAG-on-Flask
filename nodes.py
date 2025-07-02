from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Only import tools for fallback
from tools import tools

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    context: str | None

def agent(state, config=None):
    print("---CALL AGENT---")
    messages = state["messages"]
    context = state.get("context")
    from tools import build_dynamic_retriever_tool
    tools = []
    if config:
        if "tools" in config:
            tools = config["tools"]
        elif "configurable" in config and "tools" in config["configurable"]:
            tools = config["configurable"]["tools"]
    print("Tools passed to agent:", tools)
    print("Agent config:", config)
    model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct").bind_tools(tools)
    response = model.invoke(messages)
    print("Agent response:", response)
    tool_called = False
    if hasattr(response, "tool_calls") and getattr(response, "tool_calls", None):
        tool_called = True
    elif hasattr(response, "additional_kwargs") and getattr(response, "additional_kwargs", {}).get("tool_calls"):
        tool_called = True
    return {"messages": [response], "context": context, "tool_called": tool_called}

# For grade_documents structured output
class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
prompt_grader = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n \
        Here is the retrieved document: \n\n {context} \n\n\n        Here is the user question: {question} \n\n        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n\n        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    input_variables=["context", "question"],
)

def grade_document(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")
    messages = state["messages"]
    context = state.get("context")
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    chain = prompt_grader | ChatGroq(model="gemma2-9b-it").with_structured_output(Grade)
    scored_result = chain.invoke({"question": question, "context": docs})
    score = None
    if isinstance(scored_result, dict):
        score = scored_result.get('binary_score', 'no')
    else:
        score = getattr(scored_result, 'binary_score', 'no')
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"

# For RAG prompt
dag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = dag_prompt | ChatGroq(model="qwen-qwq-32b") | StrOutputParser()

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    context = state.get("context")
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content if last_message.content else context
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response], "context": context}

def rewrite(state):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    context = state.get("context")
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n \
            Look at the input and try to reason about the underlying semantic intent / meaning. \n \
            Here is the initial question:\n    \n ------- \n\n    {question} \n    \n ------- \n\n    Formulate an improved question: """,
        )
    ]
    response = ChatGroq(model="llama-3.3-70b-versatile").invoke(msg)
    return {"messages": [response], "context": context} 