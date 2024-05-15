#OpenML RAG

This project leverages an existing LLM (here Llama2) and a retrieval-augmented generation (RAG) model to provide a web API that translates user queries to an OpenML understandable vocabulary. 

The key steps are:

- an existing set of user queries to OpenML vocabulary translations is provided that is used to create vector embeddings and a retrieval index.
- When a new user query is received, the query is encoded into a vector embedding and similar embeddings are retrieved from the index.
- These retrieved examples are then used to condition the LLM to generate a response that translates the user query into OpenML vocabulary.

### Prerequisites

A running Ollama server is expected to run. To install and run the LLM:

```
curl https://ollama.ai/install.sh | sh
ollama serve
```

### install
install the project dependencies:

`make install`

### run
To run the project, execute the following command. A webserver is started that listens on `http://localhost:5000` and provides the API endpoints. 


`make run`

interact with the API by sending a POST request to the /chat endpoint with a JSON payload containing the question:
```
curl -d '{"question":"Calculate the density of the objects"}' -H "Content-Type: application/json" -X POST http://localhost:5000/chat
```