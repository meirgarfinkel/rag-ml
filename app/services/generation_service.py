import logging
from typing import List, Dict, Any

import openai
from app.core.config import Settings

logger = logging.getLogger(__name__)


MAX_CONTEXTS = 4
DEFAULT_MODEL = "gpt-4o-mini"


def build_rag_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build a RAG prompt from query and retrieved contexts.
    """
    context_texts = [
        ctx.get("text", "").strip()
        for ctx in contexts[:MAX_CONTEXTS]
        if ctx.get("text")
    ]

    context_str = "\n\n".join(context_texts)

    return f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
        If the context doesn't contain enough information, say "I don't have enough information to answer this."

        Context:
        {context_str}

        Question: {query}

        Answer:"""


def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    settings: Settings,
) -> str:
    """
    Generate an answer using an OpenAI chat completion model.
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to generate_answer")
        return "I don't have enough information to answer this."

    if not contexts:
        logger.info("No context provided to generation step")

    logger.info(f"Generating answer for query: '{query[:50]}...'")

    prompt = build_rag_prompt(query, contexts)

    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()

        logger.info(f"Generated answer ({len(answer)} chars)")
        return answer

    except Exception:
        logger.exception("OpenAI API error during answer generation")
        return "Sorry, I encountered an error while generating the answer."
