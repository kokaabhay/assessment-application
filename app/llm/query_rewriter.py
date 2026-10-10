# Necessary imports
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

from config import (
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)

client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


def rewrite_query(query: str) -> str:
    """
    Rewrite a user's question into a concise search query
    optimized for knowledge-base retrieval.
    """
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite customer support questions for "
                    "knowledge-base search. "
                    "Return only the rewritten search query. "
                    "Preserve the user's intent and important terms. "
                    "Do not answer the question."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query


def get_rewritten_query(query: str) -> str:
    try:
        return rewrite_query(query)
    except Exception as e:
        logger.error(str(e))
        logger.error("Query Re-Writing has Failed probably due to LLM API failure:")
        logger.error("=" * 60)
        logger.info(
            "Proceeding with empty Re-Written query and user query will be used as default"
        )
        logger.info("=" * 60)
        return ""
