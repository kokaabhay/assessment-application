# Necessary imports
import certifi
import httpx
import logging
logger = logging.getLogger(__name__)
from config import (
    AZURE_RERANK_ENDPOINT,
    AZURE_RERANK_API_KEY,
    AZURE_RERANK_DEPLOYMENT,
)

# create http client
http_client = httpx.Client(
    verify=False,
    timeout=60.0,
)

# print("Reranker endpoint:", repr(AZURE_RERANK_ENDPOINT))


# This function uses cohere reranker for reranking candidate documents sent by hybrid search an hyde retrieval
def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:

    if not documents:
        return []

    payload = {
        "model": AZURE_RERANK_DEPLOYMENT,
        "query": query,
        "documents": [document["content"] for document in documents],
        "top_n": top_k,
    }
    try:
        # raise
        response = http_client.post(
            AZURE_RERANK_ENDPOINT,
            headers={
                "Authorization": f"Bearer {AZURE_RERANK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()

        result = response.json()

        reranked_documents = []

        for item in result["results"]:
            document = documents[item["index"]].copy()
            document["rerank_score"] = item["relevance_score"]
            # for i in document.keys():
            #     print(i)

            reranked_documents.append(document)
            # for i in reranked_documents:
            #     print(i)
        return reranked_documents
    except Exception as e:
        logger.error(str(e))
        logger.error("Reranker not responding probably due to LLM API failure:")
        logger.error("=" * 60)
        logger.info("Proceeding to build prompt phase without reranking documents")
        logger.info("=" * 60)

        for document in documents:
            document["rerank_score"] = None

        return documents
