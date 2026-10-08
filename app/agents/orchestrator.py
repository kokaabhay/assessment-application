# Necessary imports
from openai import AzureOpenAI
import json
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


# Orchestrator class
class Orchestrator:
    def __init__(self):
        pass

    # This function defines the prompt template for orchestration
    def build_orchestrator_prompt(self, user_query: str, rewritten_query: str) -> str:
        system_prompt = f"""
                You are the Model Router agentic AI system.
                Your job is NOT to answer the user's question directly.
                Your job is to decide which knowledge base should be used
                to retrieve information for answering the user's question.
                You will be given the user's query and then the rewritten query as well.
                The rewritten query is a re-written query by the "LLm" of the original user's query
                
                
        
                Return ONLY valid JSON.
        
                The JSON format must be:
        
                {{
                    "deployment": "gpt-5-nano",
                    "reason": "Explain why this model was selected."
                }}

                OR

                {{
                    "deployment": "gpt-4.1",
                    "reason": "Explain why this document was selected."
                        }}
        
                The "documents" field MUST contain exactly one of:
        
                "gpt-4.1"
                "gpt-5-nano"
                Do not answer the user's question.
        
                user query:
                {user_query}
                Re-written user query:
                {rewritten_query}
                """
        return system_prompt

    # This function will decide which of the 2 knowledge-bases (Documents) or both should be included for retrieval based on the user and rewitten-query
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
            logger.error(str(e))
            logger.error("Orchestrator not responding probably due to LLM API failure:")
            logger.error("=" * 60)
            logger.info(
                "Proceeding to build retrieval phase with both retrieval documents in pipeline"
            )
            logger.info("=" * 60)
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
