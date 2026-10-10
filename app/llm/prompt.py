# Necessary imports
import tiktoken
from typing import Optional
from config import MAX_CONTEXT_TOKENS
from langchain_core.prompts import PromptTemplate

encoding = tiktoken.get_encoding("cl100k_base")


def build_context(documents: list[dict]) -> list[dict]:

    selected_documents = []
    total_tokens = 0

    for document in documents:

        content = document.get("content", "").strip()

        if not content:
            continue

        token_count = len(encoding.encode(content))

        if total_tokens + token_count > MAX_CONTEXT_TOKENS:
            break

        selected_documents.append(document)
        total_tokens += token_count
    return selected_documents


PROMPT_TEMPLATE = PromptTemplate.from_template("""
You are a helpful HR Policy & Employee Handbook Assistant.

Answer the user's question using ONLY the information
provided in the knowledge base context below.

If the answer cannot be found in the knowledge base,
say that you do not have enough information to answer
the question.

Be concise, polite, and easy to understand.

If the user question doesn't need any retrieval, then the
Knowledge Base Context and the HyDE Retrieved Knowledge
Base Context will be empty.

Knowledge Base Context:
------------------------
{context}
------------------------
HyDE Retrieved Knowledge Base Context:
------------------------
{h_context}
------------------------

User Question:
{query}

Instructions:
- Do not invent information.
- Do not use outside knowledge.
- Answer directly.
- When possible, mention the source document on a new line as well.
""".strip())


def build_prompt(
    query: str,
    documents: list[dict] | None,
    h_documents: list[dict] | None,
) -> str:

    context_parts = []
    hyde_context_parts = []

    if documents:
        for i, document in enumerate(documents, start=1):
            context_parts.append(
                f"[Source {i}: {document['source']}]\n" f"{document['content']}"
            )

    if h_documents:
        for i, document in enumerate(h_documents, start=1):
            hyde_context_parts.append(
                f"[Source {i}: {document['source']}]\n" f"{document['content']}"
            )

    context = "\n\n".join(context_parts)
    h_context = "\n\n".join(hyde_context_parts)

    return PROMPT_TEMPLATE.format(
        context=context,
        h_context=h_context,
        query=query,
    )
