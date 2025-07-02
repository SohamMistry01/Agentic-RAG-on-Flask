from flask import Flask, render_template, request
from graph import graph
from tools import extract_text_from_url, build_dynamic_retriever_tool
import markdown as md
from markupsafe import Markup
import io
import sys
from contextlib import redirect_stdout
from langchain_core.messages import HumanMessage

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    user_query = None
    url = None
    tool_called = None
    debug_output = None
    if request.method == 'POST':
        url = request.form.get('url')
        user_query = request.form.get('query')
        if url and user_query:
            try:
                retriever_tool = build_dynamic_retriever_tool(url)
                url_text = extract_text_from_url(url)
                f = io.StringIO()
                with redirect_stdout(f):
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_query)], "context": url_text},
                        config={"tools": [retriever_tool]}
                    )
                debug_output = f.getvalue()
                messages = result.get('messages', [])
                tool_called = result.get('tool_called', None)
                if messages:
                    if isinstance(messages[-1], str):
                        response = messages[-1]
                    else:
                        response = getattr(messages[-1], 'content', str(messages[-1]))
            except Exception as e:
                response = f"Error processing URL: {e}"
    return render_template('index.html', response=response, user_query=user_query, url=url, tool_called=tool_called, debug_output=debug_output)

@app.template_filter('markdown')
def markdown_filter(text):
    if not text:
        return ""
    return Markup(md.markdown(text, extensions=['fenced_code', 'tables']))

if __name__ == '__main__':
    app.run(debug=True) 