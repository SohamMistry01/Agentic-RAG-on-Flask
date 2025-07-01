from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field
from tools import tools

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# Instantiate models and prompt once for reuse
chat_model = ChatGroq(model="llama-3.3-70b-versatile")
chat_model_with_tools = chat_model.bind_tools(tools)

# For grade_documents structured output
class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")
llm_with_tool = chat_model.with_structured_output(Grade)
prompt_grader = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n \
        Here is the retrieved document: \n\n {context} \n\n\n        Here is the user question: {question} \n\n        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n\n        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    input_variables=["context", "question"],
)

# For RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | chat_model | StrOutputParser()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def agent(state):
    messages = state["messages"]
    response = chat_model_with_tools.invoke(messages)
    return {"messages": [response]}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    chain = prompt_grader | llm_with_tool
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    if score == "yes":
        return "generate"
    else:
        return "rewrite"

def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state):
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f""" \n \
    Look at the input and try to reason about the underlying semantic intent / meaning. \n \
    Here is the initial question:\n    \n ------- \n\n    {question} \n    \n ------- \n\n    Formulate an improved question: """,
        )
    ]
    response = chat_model.invoke(msg)
    return {"messages": [response]} 