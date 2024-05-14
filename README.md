#OpenML RAG

### install
make install

### run
make run

interact with the API by sending a POST request to the /chat endpoint with a JSON payload containing the question:
```
curl -d '{"question":"Calculate the density of the objects"}' -H "Content-Type: application/json" -X POST http://localhost:5000/chat
```