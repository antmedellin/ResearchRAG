# Local ResearcherRAG

This system is designed to run locally on your machine with a local LLM and vector database. It creates a knowledge base from your documents and allows you to ask questions about them. It uses a hybrid approach of vector search and BM25 search to find relevant documents, and then uses a local LLM to generate answers.


veify can read .env file
docker compose --env-file .env config

pull docker images
docker compose --env-file .env pull

start docker containers
docker compose --env-file .env up -d


test embedding endpoint
curl http://localhost:11434/api/embeddings -d '{
  "model": "mxbai-embed-large:335m",
  "prompt": "Represent this sentence for searching relevant passages: The sky is blue because of Rayleigh scattering"
}'

test decision endpoint
curl http://localhost:11434/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef-flash:9b",
    "state": "Howdy World",
    "questions": {
      "says_hello": {
        "type": "noul",
        "instructions": "Does the state text contain a greeting?",
        "criteria": {
          "true": "The state text contains a greeting.",
          "false": "The state text does not contain a greeting."
        }
      }
    }
  }'

test chat endpoint
curl -i http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:12b",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'