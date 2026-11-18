#!/usr/bin/env python3
"""Insert five sample paragraphs into a Qdrant test collection.

Each dense embedding is requested separately from the OpenAI-compatible URL
in .env. Each paragraph is also encoded separately with FastEmbed BM42.
"""

from __future__ import annotations

import os
import sys
import uuid

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

PARAGRAPHS = [
    "A red fox moved quietly through the snowy forest at dawn. Its thick winter coat helped it stay warm while it searched beneath the trees for small animals.",
    "Solar panels convert sunlight into electricity. Their output depends on panel orientation, shading, weather, and the efficiency of the photovoltaic cells.",
    "A suspension bridge carries traffic across a wide river using cables anchored to towers. The main cables support vertical hangers, which hold the roadway deck.",
    "Coffee beans develop different flavors during roasting. Lighter roasts often retain more of the bean's original acidity, while darker roasts tend to have stronger toasted notes.",
    "Coral reefs support many marine species and protect coastlines from wave energy. Rising ocean temperatures can cause coral bleaching and threaten reef ecosystems.",
]

DENSE_VECTOR_NAME = "default"
SPARSE_VECTOR_NAME = "bm42"
SPARSE_MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"


def dense_embed_one(url: str, model: str, text: str, api_key: str | None) -> list[float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        url,
        headers=headers,
        json={"model": model, "input": [text]},
        timeout=(15, 600),
    )
    response.raise_for_status()
    records = response.json().get("data")
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError(f"Expected one OpenAI-compatible embedding result; received: {response.text}")
    vector = records[0].get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"Embedding response has no nonempty 'embedding' array: {response.text}")
    return [float(value) for value in vector]


def sparse_embed_one(model: SparseTextEmbedding, text: str) -> models.SparseVector:
    embedding = next(model.embed([text]))
    return models.SparseVector(
        indices=[int(index) for index in embedding.indices],
        values=[float(value) for value in embedding.values],
    )


def ensure_test_collection(client: QdrantClient, name: str, dimension: int) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dimension,
                    distance=models.Distance.DOT,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        print(f"Created collection '{name}' with dense dimension {dimension} and BM42 sparse vectors.")
        return

    info = client.get_collection(name)
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
        raise RuntimeError(f"Existing collection '{name}' must have named dense vector '{DENSE_VECTOR_NAME}'.")
    if vectors[DENSE_VECTOR_NAME].size != dimension:
        raise RuntimeError(
            f"Existing collection '{name}' has dimension {vectors[DENSE_VECTOR_NAME].size}; "
            f"the configured embedding model returned {dimension}. Choose another test collection."
        )
    sparse = info.config.params.sparse_vectors
    if not isinstance(sparse, dict) or SPARSE_VECTOR_NAME not in sparse:
        raise RuntimeError(f"Existing collection '{name}' is missing sparse vector '{SPARSE_VECTOR_NAME}'.")


def main() -> int:
    load_dotenv()
    embedding_url = os.getenv("DENSE_EMBEDDING_URL", "").strip()
    embedding_model = os.getenv("DENSE_EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "")).strip()
    api_key = os.getenv("DENSE_EMBEDDING_API_KEY", "").strip() or None
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    collection = os.getenv("QDRANT_TEST_COLLECTION", "researcher_rag_smoke_test").strip()

    if not embedding_url:
        print("Set DENSE_EMBEDDING_URL in .env to your OpenAI-compatible /v1/embeddings URL.", file=sys.stderr)
        return 2
    if not embedding_model:
        print("Set DENSE_EMBEDDING_MODEL (or EMBED_MODEL) in .env.", file=sys.stderr)
        return 2

    client = QdrantClient(url=qdrant_url)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    point_ids: list[str] = []
    expected_dimension: int | None = None

    for index, paragraph in enumerate(PARAGRAPHS, start=1):
        # One API request per paragraph, deliberately avoiding a batch request.
        dense_vector = dense_embed_one(embedding_url, embedding_model, paragraph, api_key)
        if expected_dimension is None:
            expected_dimension = len(dense_vector)
            ensure_test_collection(client, collection, expected_dimension)
        elif len(dense_vector) != expected_dimension:
            raise RuntimeError(
                f"Paragraph {index} returned dimension {len(dense_vector)}; expected {expected_dimension}."
            )

        sparse_vector = sparse_embed_one(sparse_model, paragraph)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant-smoke-test:{index}"))
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={DENSE_VECTOR_NAME: dense_vector, SPARSE_VECTOR_NAME: sparse_vector},
                    payload={
                        "text": paragraph,
                        "metadata": {
                            "title": f"Smoke test paragraph {index}",
                            "paragraph_number": index,
                            "source": "qdrant_smoke_test.py",
                        },
                    },
                )
            ],
            wait=True,
        )
        point_ids.append(point_id)
        print(f"Embedded and inserted paragraph {index}/5 ({len(dense_vector)} dense values; {len(sparse_vector.indices)} sparse values).")

    count = client.count(collection_name=collection, exact=True).count
    print(f"\nSuccess: inserted 5 paragraphs into '{collection}'. Collection now contains {count} points.")
    print(f"Qdrant URL: {qdrant_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
