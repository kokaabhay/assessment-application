# Necessary imports
from openai import AzureOpenAI
import json
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)
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

AGENTS = {
    "rag_agent": {
        "description": "Answers questions using the company's knowledge base."
        # },
        # "order_agent": {
        #     "description": "Checks order status and order information."
        # },
        # "support_agent": {
        #     "description": "Handles general customer support conversations."
        # }
    }
}


# Orchestrator class
class Orchestrator:
    def __init__(self):
        pass

    # This function defines the prompt template for orchestration
    def build_orchestrator_prompt(self, user_query: str, rewritten_query: str) -> str:
        system_prompt = f"""
                You are the orchestrator of a customer-support agentic AI system.
                Your job is NOT to answer the user's question directly.
                Your job is to decide which knowledge base should be used
                to retrieve information for answering the user's question.
                You will be given the user's query and then the rewritten query as well.
                The rewritten query is a re-written query by the "LLm" of the original user's query
        
                There are two knowledge bases.
        
                DOCUMENT 1:
                This document provides customer support representatives with
                information and standard responses for common customer questions
                and issues.
        
                It primarily covers:
                - hardware
                - Wi-Fi
                - device pairing
                - common troubleshooting
                - standard customer support issues
        
                DOCUMENT 2:
                This document focuses on:
                - account management
                - household access
                - service configuration
                - device ownership
                - notifications
                - customer data requests
        
                It is intended for situations that are different from
                hardware, Wi-Fi, and device-pairing troubleshooting.
        
                Your task is to determine whether the user's question requires
                information from document 1, document 2, or both.
        
                Return ONLY valid JSON.
        
                The JSON format must be:
        
                {{
                    "documents": "1",
                    "reason": "Explain why this document was selected."
                }}
        
                The "documents" field MUST contain exactly one of:
        
                "1"
                "2"
                "both"
        
                Do not answer the user's question.
        
                user query:
                {user_query}
                Re-written user query:
                {rewritten_query}
                """
        return system_prompt

    # This function will decide which of the 2 knowledge-bases (Documents) or both should be included for retrieval based on the user and rewitten-query
    # Tries 1 times with a delay of 10 seconds between each attempt
    # @retry(stop_after_attempt(1),wait_fixed(10))
    def orchestrate(self, user_query: str, rewritten_query: str) -> dict:
        tools = []
        system_prompt = self.build_orchestrator_prompt(user_query, rewritten_query)
        response = client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0,
            tools=tools,
        )
        decision = json.loads(response.choices[0].message.content)
        return decision

    # This function will call the orchestrate function and in case of failure ensures a working fallback
    def get_decision(self, user_query: str, rewritten_query: str) -> dict:
        try:
            return self.orchestrate(user_query, rewritten_query)
        except Exception as e:
            print(str(e))
            print("Orchestrator not responding probably due to LLM API failure: \n")
            print("=" * 60, "\n")
            print(
                "Proceeding to build retrieval phase with both retrieval documents in pipeline"
            )
            print("=" * 60)
            return {
                "documents": "both",
                "reason": "The orchestrator failed so considering both documents for maximum context",
            }


# """ from openai import AzureOpenAI
# import json

# client = AzureOpenAI(
#     api_key="AZURE_API_KEY",
#     api_version="2024-10-21",
#     azure_endpoint="AZURE_ENDPOINT"
# )

# DEPLOYMENT = "gpt-4.1"


# AGENTS = {
#     "rag_agent": {
#         "description": "Answers questions using the company's knowledge base."
#     },
#     "order_agent": {
#         "description": "Checks order status and order information."
#     },
#     "support_agent": {
#         "description": "Handles general customer support conversations."
#     }
# }


# def orchestrate(user_query: str):

#     system_prompt = f"""
# You are the orchestrator of a customer-support agentic AI system.

# Your job is NOT to answer the user's question directly.

# Your job is to decide which agent should handle the request.

# Available agents:

# {json.dumps(AGENTS, indent=2)}

# Return ONLY valid JSON in this format:

# {{
#     "agent": "rag_agent",
#     "reason": "The user is asking about information contained in the knowledge base."
# }}

# Choose exactly one agent.

# User query:
# {user_query}
# """

#     response = client.chat.completions.create(
#         model=DEPLOYMENT,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             }
#         ],
#         temperature=0
#     )

#     decision = json.loads(response.choices[0].message.content)

#     return decision """
