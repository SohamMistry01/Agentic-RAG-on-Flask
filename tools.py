import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
import requests
from bs4 import BeautifulSoup

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is not None:
    os.environ['HF_TOKEN'] = hf_token

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def extract_text_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text

def build_dynamic_retriever_tool(url):
    text = extract_text_from_url(url)
    doc = Document(page_content=text, metadata={"source": url})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_splits = text_splitter.split_documents([doc])
    vectorstore = FAISS.from_documents(documents=docs_splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="Blog_Retriever",
        description="ALWAYS use this tool to answer any question about the provided URL. This tool searches the content of the URL."
    )
    return retriever_tool

tools = [build_dynamic_retriever_tool]