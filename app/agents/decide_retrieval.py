# Necessary imports
from openai import OpenAI
from config import (
    AZURE_CHAT_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)
from tenacity import retry, wait_fixed, stop_after_attempt

# create the client to talk to your Azure OpenAI resource.
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


# This function returns a bool True/False whether retrieval is necessary or not required on the first place based on the user query
# Tries 1 times with a delay of 10 seconds between each attempt
# @retry(stop_after_attempt(1),wait_fixed(10))
def decide_retrieve(prompt: str) -> bool:
    response = client.chat.completions.create(
        model=AZURE_CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    The user will ask a query and it will be forwarded to the HR Policy & Employee Handbook Assistant agent bot
                    The usecase is to answer from upload policy documents, employee handbooks, and benefits guides.
                    for example:-
                    user:Good morning,Hello-> not needed
                    user:what are the flexible work options I have?-> the bot needs to get context from documents to answer the question correctly
                    Remember the question is supposed to be relevant to the HR policy and employees in a company
                    Return False if retrieval is not needed and True if it is needed based on the user query"""
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip()
    return result == "True"


# This function will call the decide_retrieve function and in case of failure ensures a working fallback
# print(type(bool(decide_retrieve("I need a product...ans also calcium is not healthy for body...i want to buy calicum"))))
def get_retrieval_decision(prompt: str) -> bool:
    try:
        return decide_retrieve(prompt)
    except Exception as e:
        print(str(e))
        print("unable to get retrieval decision probably due to LLM API failure  : \n")
        print("=" * 60, "\n")
        print("Proceeding initiate retrieval process by default")
        print("=" * 60)
        return True
