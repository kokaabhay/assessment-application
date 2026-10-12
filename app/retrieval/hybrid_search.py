# Necessary imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import logging

logger = logging.getLogger(__name__)
from openai import OpenAI
from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_EMBEDDING_DEPLOYMENT,
)

SOURCE_FILE_MAPPING = {
    "policy_documents/3ce91453-506b-4595-b4eb-06b1b5eaeef8.txt": "03_Employee_Benefits_Insurance_Guide.pdf",
    "policy_documents/22ed141c-4e7f-426b-bd33-a38b2520817d.txt": "01_Employee_Leave_Time_Off_Handbook.docx",
    "policy_documents/a7c1253f-afa4-4acf-825c-d6b2715f09c6.txt": "02_Remote_Flexible_Work_Policy.txt",
    "policy_documents/8f2f898f-acc6-486f-b2c8-c5da21dfd984.txt": "05_HR_Policy_FAQ_Knowledge_Base.csv",
    "policy_documents/287679bd-fe7c-427f-bd68-26ade8c16f3c.txt": "04_Workplace_Conduct_Anti_Harassment_Training.pdf",
}


search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)


embedding_client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


def generate_query_embedding(query: str) -> list[float]:
    response = embedding_client.embeddings.create(
        model=AZURE_EMBEDDING_DEPLOYMENT,
        input=query,
        dimensions=1536,
    )
    return response.data[0].embedding


def format_search_results(results):
    documents = []
    for result in results:
        source_file = result.get("source_file", "")
        documents.append(
            {
                "content": result.get("content", ""),
                "source": SOURCE_FILE_MAPPING.get(source_file, source_file),
                "source_file": source_file,
                "chunk_id": result.get("chunk_id", ""),
                "document_id": result.get("document_id", ""),
                "document_type": result.get("document_type", ""),
                "category": result.get("category", ""),
                "section": result.get("section", ""),
                "chunk_index": result.get("chunk_index", 0),
                "score": result.get("@search.score", 0),
                "rerank_score": None,
            }
        )
    return documents


def hybrid_search(
    query: str,
    rewritten_query,
    top_k: int = 5,
):
    query_vector = (
        generate_query_embedding(rewritten_query)
        if rewritten_query
        else generate_query_embedding(query)
    )

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
        exhaustive=True,
    )

    results = search_client.search(
        search_text=rewritten_query if rewritten_query else query,
        vector_queries=[vector_query],
        top=top_k,
        select=[
            "chunk_id",
            "document_id",
            "source_file",
            "content",
            "document_type",
            "category",
            "section",
            "chunk_index",
        ],
    )

    return format_search_results(results)


def hyde_retrieval(hyde_answer: str, top_k: int = 5):
    hyde_vector = generate_query_embedding(hyde_answer)

    vector_query = VectorizedQuery(
        vector=hyde_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
        exhaustive=True,
    )

    results = search_client.search(
        vector_queries=[vector_query],
        top=top_k,
        select=[
            "chunk_id",
            "document_id",
            "source_file",
            "content",
            "document_type",
            "category",
            "section",
            "chunk_index",
        ],
    )

    return format_search_results(results)
