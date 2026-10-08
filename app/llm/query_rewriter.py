# Necessary imports
from openai import OpenAI
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from config import (
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)

# create the client to talk to your Azure OpenAI resource.
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


# This function rewites the user query for better vector / keyword based retrieval
# Tries 1 times with a delay of 10 seconds between each attempt
# @retry(stop_after_attempt(1),wait_fixed(10))
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


# This function will call the rewrite_query function and in case of failure ensures a working fallback
def get_rewritten_query(query: str) -> str:
    try:
        return rewrite_query(query)
    except Exception as e:
        print(str(e))
        print("Query Re-Writing has Failed probably due to LLM API failure: \n")
        print("=" * 60, "\n")
        print(
            "Proceeding with empty Re-Written query and user query will be used as default"
        )
        print("=" * 60)
        return ""
