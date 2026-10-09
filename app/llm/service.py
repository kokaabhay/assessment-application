# Necessary imports
from openai import OpenAI
from config import (
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)
import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)
# create the client to talk to your Azure OpenAI resource.
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


# Generates the final LLM Response to be replied to the end user asking the question
def generate_answer(prompt: str, d) -> str:

    response = client.chat.completions.create(
        model=d,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an HR Policy & Employee Handbook Assistant  "
                    "that answers using a provided knowledge base."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# This function will call the generate_answer function and in case of failure ensures a working fallback
def get_answer(prompt: str, d) -> str:
    try:
        return generate_answer(prompt, d)

    except Exception as e:
        logger.error(str(e))
        logger.error("LLM not responding probably due to LLM API failure:")
        logger.error("=" * 60)
        logger.info("Proceeding with default response to user")
        logger.info("=" * 60)

        return (
            "OOOPS! Sorry, our  support agent is currently "
            "unavailable at the moment.\n\n"            
            "We apologise for the inconvenience caused."
        )
