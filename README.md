# Local ResearcherRAG

Faculty research portfolio from Digital Measures -> Neo4j (graph) + Qdrant (hybrid BM25 + dense search).
Departments are loaded one at a time; shared people/publications/grants are deduplicated across departments.



veify can read .env file
docker compose --env-file .env config

pull docker images
docker compose --env-file .env pull

start docker containers
docker compose --env-file .env up -d

recreate docker container
docker compose up -d --force-recreate user_app







