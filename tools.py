from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is not None:
    os.environ['HF_TOKEN'] = hf_token

# Load HuggingFace embeddings only once
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# LangGraph URLs
urls=[
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/"
]
docs=[WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)
vectorstore = FAISS.from_documents(documents=docs_splits, 
                                   embedding = embedding_model)
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "Blog_Retriever",
    "Search and Run information about LangGraph"
)

# LangChain URLs
langchain_urls=[
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]
docs2 = [WebBaseLoader(url).load() for url in langchain_urls]
docs_list2 = [item for sublist in docs2 for item in sublist]
doc_splits = text_splitter.split_documents(docs_list2)
vectorstorelangchain=FAISS.from_documents(documents=doc_splits, 
                                          embedding = embedding_model)
retrieverlangchain=vectorstorelangchain.as_retriever()
retriever_tool_langchain=create_retriever_tool(
    retrieverlangchain,
    "LangChain_blog_retriever",
    "Search and run information about Langchain"
)

tools = [retriever_tool, retriever_tool_langchain] 