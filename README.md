# Agentic RAG: Know About LangChain & LangGraph

A modern Flask web application demonstrating an Agentic Retrieval-Augmented Generation (RAG) pipeline using LangChain, LangGraph, HuggingFace Embeddings, and Groq LLMs. This project allows users to ask questions about LangChain and LangGraph, retrieving and generating answers with advanced agentic workflows.

## Features
- **Agentic RAG Pipeline**: Combines retrieval and generation with agent-based decision logic.
- **LangChain & LangGraph**: Utilizes state-of-the-art libraries for composable LLM workflows.
- **HuggingFace Embeddings**: For semantic document retrieval.
- **Groq LLM Integration**: Fast, high-quality language model responses.
- **Modern Dark-Themed UI**: User-friendly and aesthetic web interface.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Agentic-RAG
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root with your API keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     HF_TOKEN=your_huggingface_token
     ```

5. **Run the Flask app**
   ```bash
   python app.py
   ```
   Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Usage
- Enter your question about LangChain or LangGraph in the input box and submit.
- The app will retrieve relevant documents, grade their relevance, and generate a response using the agentic workflow.

## Project Structure
```
Agentic-RAG/
  app.py            # Flask app entry point
  graph.py          # LangGraph workflow setup
  nodes.py          # Agent, grader, generator, and rewriter logic
  tools.py          # Document loaders, retrievers, and embeddings
  requirements.txt  # Python dependencies
  static/           # CSS and static assets
  templates/        # HTML templates
```

## Credits
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [HuggingFace](https://huggingface.co/)
- [Groq](https://groq.com/)

---
*Built with ❤️ for the agentic AI community.* 