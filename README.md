# Agentic RAG: Dynamic Agentic Retrieval-Augmented Generation

A modern Flask web application demonstrating an Agentic Retrieval-Augmented Generation (RAG) pipeline using LangChain, LangGraph, HuggingFace Embeddings, and Groq LLMs. This project allows users to ask questions about **any URL**—the app dynamically loads, embeds, and retrieves from the provided web page, then uses an agentic workflow to generate answers with step-by-step logic.

## Features
- **Dynamic URL RAG**: Ask questions about any web page by providing its URL.
- **Agentic Workflow**: Combines retrieval, grading, rewriting, and generation with agent-based decision logic (see debug trace in terminal).
- **LangChain & LangGraph**: State-of-the-art libraries for composable LLM workflows.
- **HuggingFace Embeddings**: For semantic document retrieval.
- **Groq LLM Integration**: Fast, high-quality language model responses.
- **Modern UI**: User-friendly, dark-themed web interface.
- **Terminal Debug Trace**: See step-by-step agentic reasoning in your terminal (e.g., `---CALL AGENT---`, `---CHECK RELEVANCE---`, etc.).

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
- Enter a URL and your question in the input boxes and submit.
- The app will:
  1. Load and embed the web page.
  2. Retrieve relevant chunks using semantic search.
  3. Use an agentic workflow to decide whether to retrieve, rewrite, or generate.
  4. Show the final answer in the UI.
- **Debug Trace**: See the step-by-step agentic reasoning in your terminal (e.g., `---CALL AGENT---`, `---CHECK RELEVANCE---`, `---GENERATE---`).

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

## Optimization Tips
- **Use a Faster Model**: Swap to a smaller or more optimized LLM (e.g., `llama-3-8b`, `gpt-3.5-turbo`) for faster responses.
- **Increase Chunk Size**: In `tools.py`, use a larger `chunk_size` and smaller `chunk_overlap` in the text splitter to reduce retrieval time.
- **Limit Retriever Results**: Set `retriever.search_kwargs = {'k': 2}` to reduce LLM context size.
- **Cache Vectorstores**: If you expect repeated queries for the same URL, cache the embeddings/vectorstore.
- **Profile Your Pipeline**: Use Python's `time` module to find bottlenecks.

## Troubleshooting
- **Only `---CALL AGENT---` Appears in Debug**: This means the agent is not receiving tools. Make sure you are passing tools via the `config` and that your `agent` node checks both `config['tools']` and `config['configurable']['tools']`.
- **Slow Responses**: Try a smaller model, increase chunk size, or limit retriever results as above.
- **Tool Not Called**: Make your tool description very explicit (e.g., "ALWAYS use this tool to answer any question about the provided URL.").

## Credits
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [HuggingFace](https://huggingface.co/)
- [Groq](https://groq.com/)

---
*Built with ❤️ for the agentic AI community.* 