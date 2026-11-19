# Local ResearcherRAG

Faculty research portfolio from Digital Measures -> Neo4j (graph) + Qdrant (hybrid BM25 + dense search).
Departments are loaded one at a time; shared people/publications/grants are deduplicated across departments.



veify can read .env file
docker compose --env-file .env config

pull docker images
docker compose --env-file .env pull

start docker containers
docker compose --env-file .env up -d


test embedding endpoint
curl http://localhost:11434/api/embed -d '{
  "model": "mxbai-embed-large:335m",
  "input": "Represent this sentence for searching relevant passages: The sky is blue because of Rayleigh scattering"
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





docker compose up -d --force-recreate user_app



# step 0 lists all departments and number of faculty in each 
docker compose exec user_app python src/research_portfolio/ledger/discover_faculty.py --config config/departments/maie.yaml --list-departments

# step 1 discover faculty in dept  
docker compose exec user_app python src/research_portfolio/ledger/discover_faculty.py \
  --config config/departments/maie.yaml

# step 2
docker compose exec user_app python src/research_portfolio/ledger/fetch_profiles.py --config config/departments/maie.yaml

# step 3
docker compose exec user_app python src/research_portfolio/ledger/build_ledger.py --config config/departments/maie.yaml

# step 4 dry run 
docker compose exec user_app python src/research_portfolio/run_department.py \
  --config config/departments/maie.yaml --skip-discover --skip-fetch

# step 5 ingest into graph qdrant
docker compose exec user_app python src/research_portfolio/run_department.py \
  --config config/departments/maie.yaml \
  --skip-discover --skip-fetch \
  --apply

# step 6 validate
docker compose exec user_app python \
  -m research_portfolio.workflow.validate_department \
  --config config/departments/ies.yaml

# ask questions / query 
ask questions 
docker compose exec user_app python src/research_portfolio/retrieval/search.py \
  "What are dr Moya research expertise?" \
  --person "Hiram Moya" \
  --ask


docker compose exec user_app python src/research_portfolio/retrieval/search.py \
  "additive manufacturing and advanced manufacturing for defense applications" \
  --faculty \
  --dept maie


docker compose exec user_app python \
  src/research_portfolio/retrieval/search.py \
  "publications" \
  --person "Hiram Moya" \
  --list-publications \
  --limit 20 \
  --since 2020 
 
docker compose exec user_app python src/research_portfolio/retrieval/search.py \
  "advanced manufacturing defense research" \
  --types grant --funded



Faculty matching should eventually be multi-signal

 Plain Text
Candidate = Hiram Moya
 
Technical relevance 0.83
Past grant relevance 0.91
Sponsor experience DHS
Recent activity 0.71
Collaboration/team fit 0.86
Evidence strength 0.89

verify openalex, vllm, lamma swap 

click start job keeps reverting to fine arts instead saying on dept