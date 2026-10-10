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

client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=AZURE_OPENAI_API_KEY,
)


# Orchestrator class
class Orchestrator:
    def __init__(self):
        pass

    def build_orchestrator_prompt(self, user_query: str, rewritten_query: str) -> str:
        system_prompt = f"""
                You are the Model Router agentic AI system.
                Your job is NOT to answer the user's question directly.
                Your job is to decide which Model should be used for
                answering the user's question.
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
                    "reason": "Explain why this Model was selected."
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

    def get_decision(self, user_query: str, rewritten_query: str) -> dict:
        try:
            return self.orchestrate(user_query, rewritten_query)
        except Exception as e:
            logger.error(str(e))
            logger.error("Orchestrator not responding probably due to LLM API failure:")
            logger.error("=" * 60)
            logger.info("Proceeding to with gpt-4.1")
            logger.info("=" * 60)
            return {
                "deployments": "gpt-4.1",
                "reason": "The orchestrator failed so considering gpt-4.1",
            }
