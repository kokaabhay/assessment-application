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


# Generates the final LLM Response to be replied to the end user asking the question
# Tries 1 times with a delay of 10 seconds between each attempt
# @retry(stop_after_attempt(1),wait_fixed(10))
def generate_answer(prompt: str) -> str:
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
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
def get_answer(prompt: str) -> str:
    try:
        return generate_answer(prompt)

    except Exception as e:
        logger.error(str(e))
        logger.error("LLM not responding probably due to LLM API failure:")
        logger.error("=" * 60)
        logger.info("Proceeding with default response to user")
        logger.info("=" * 60)

        return (
            "OOOPS! Sorry, our customer support agent is currently "
            "unavailable at the moment.\n\n"
            "To talk to SmartHome Hub customer support, please dial "
            "to +91 xxxxxxxxxx or mail us at "
            "customersupport@smarthome.com. "
            "We apologise for the inconvenience caused."
        )
