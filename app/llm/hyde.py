# Necessary imports
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from config import (
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)
import logging
logger = logging.getLogger(__name__)
# create the client to talk to your Azure OpenAI resource.
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


# Generate a hypothetical answer and then use this answer for better quality retrieval only based on vector search
# Tries 1 times with a delay of 10 seconds between each attempt
# @retry(stop_after_attempt(1),wait_fixed(10))
def generate_hypothetical_answer(rewritten_query: str, query: str) -> str:
    """
    Generate a hypothetical answer to the user's question into a concise search reference
    optimized for semantic based knowledge-base retrieval.
    """
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a concise hypothetical answer that represents "
                    "the information likely to be found in a customer-support "
                    "knowledge base. Include important technical terms and "
                    "concepts from the user's question. "
                    "Do not mention that the answer is hypothetical. "
                    "Do not add unrelated information. "
                    "Return only the hypothetical answer."
                ),
            },
            {
                "role": "user",
                "content": query if not rewritten_query else rewritten_query,
            },
        ],
        temperature=0,
    )
    hypothetical_answer = response.choices[0].message.content.strip()
    return hypothetical_answer


# This function will call the generate_hypothetical_answer function and in case of failure ensures a working fallback
def get_hypothetical_answer(rewritten_query: str, query) -> str:
    try:
        return generate_hypothetical_answer(rewritten_query, query)
    except Exception as e:
        logger.error(str(e))
        logger.error(
            "Hypothetical answer generator not responding probably due to LLM API failure:"
        )
        logger.error("=" * 60)
        logger.info("Proceeding without hypothetical answer")
        logger.info("=" * 60)
        return None
