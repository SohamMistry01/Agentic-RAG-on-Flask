from flask import Flask, render_template, request
from graph import graph

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    user_query = None
    if request.method == 'POST':
        user_query = request.form.get('query')
        if user_query:
            # Run the graph with the user query
            result = graph.invoke({"messages": user_query})
            # Extract the answer from the result
            messages = result.get('messages', [])
            if messages:
                # If message is a string, use it directly; else, try .content
                if isinstance(messages[-1], str):
                    response = messages[-1]
                else:
                    response = getattr(messages[-1], 'content', str(messages[-1]))
    return render_template('index.html', response=response, user_query=user_query)

if __name__ == '__main__':
    app.run(debug=True) 