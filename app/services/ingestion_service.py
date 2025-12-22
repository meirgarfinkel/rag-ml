import logging
import re
from typing import List, Dict, Literal

from app.core.config import settings
from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import save_vector_store, VectorStore

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 520
CHUNK_SIZE = 600
MAX_CHUNK_SIZE = 680
OVERLAPP_SIZE = 80


def chunk_text(text: str) -> List[str]:
    """
    Sentence-aware text chunker with overlap and size constraints.

    Invariants:
    - No chunk exceeds MAX_CHUNK_SIZE
    - Chunks aim for CHUNK_SIZE when possible
    - Chunks smaller than MIN_CHUNK_SIZE are avoided unless forced
    - Overlap is preserved between adjacent chunks
    - Word boundaries are respected unless unavoidable

    Returns:
        List[str]: Ordered list of chunks
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []

    current = ""
    prev_overlap = ""

    for sentence in sentences:
        if not sentence:
            continue

        while sentence:
            combined_len = len(current) + (1 if current else 0) + len(sentence)

            # Case 1: Fits cleanly under soft target
            if combined_len <= CHUNK_SIZE:
                current = f"{current} {sentence}".strip() if current else sentence
                sentence = ""

            # Case 2: Fits under hard limit → finalize chunk
            elif combined_len <= MAX_CHUNK_SIZE:
                current = f"{current} {sentence}".strip() if current else sentence
                chunks.append(current)
                prev_overlap = current[-OVERLAPP_SIZE:]
                current = prev_overlap
                sentence = ""

            # Case 3: Current is already useful → emit it
            elif len(current) >= MIN_CHUNK_SIZE:
                chunks.append(current.strip())
                prev_overlap = current[-OVERLAPP_SIZE:]
                current = prev_overlap

            # Case 4: Forced split (sentence too large, current too small)
            else:
                sentence = f"{current} {sentence}".strip() if current else sentence
                split_pos = sentence.rfind(" ", 0, CHUNK_SIZE)

                # Unavoidable hard split
                if split_pos == -1:
                    split_pos = CHUNK_SIZE

                part, sentence = sentence[:split_pos].strip(), sentence[split_pos:].strip()
                chunks.append(part)
                prev_overlap = part[-OVERLAPP_SIZE:]
                current = prev_overlap

                if len(current) + len(sentence) < MIN_CHUNK_SIZE:
                    current = f"{current} {sentence}".strip()
                    sentence = ""
    if current:
        chunks.append(current.strip())

    return chunks


# -------------------------------------------------
# Ingestion pipeline
# -------------------------------------------------


def process_text(
    text: str,
    doc_id: str,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore
) -> Dict[str, int]:
    """
    Chunk text, embed chunks, and store vectors.
    """

    text = text.strip()
    if not text:
        return {
            "chunks_added": 0,
            "total_docs": vector_store.ntotal,
        }

    logger.info(
        "Processing document '%s' (%d chars) using %s chunking",
        doc_id,
        len(text),
    )

    # Step 1: Chunk
    chunks = chunk_text(text)
    logger.info("Created %d chunks", len(chunks))

    if not chunks:
        return {
            "chunks_added": 0,
            "total_docs": vector_store.ntotal,
        }

    # Step 2: Embed
    embeddings = embedding_model.embed_batch(chunks)
    # Pair chunks with embeddings safely
    valid_pairs = [
        (chunk, emb)
        for chunk, emb in zip(chunks, embeddings)
        if emb is not None and len(emb) > 0
    ]

    if not valid_pairs:
        logger.warning("No valid embeddings generated for document '%s'", doc_id)
        return {
            "chunks_added": 0,
            "total_docs": vector_store.ntotal,
        }

    chunks, embeddings = zip(*valid_pairs)
    chunks = list(chunks)
    embeddings = list(embeddings)

    # Step 3: Metadata
    metadatas = [
        {
            "text": chunk,
            "doc_id": doc_id,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    # Step 4: Store
    vector_store.add_embeddings(embeddings, metadatas)
    save_vector_store(vector_store, settings)

    return {
        "chunks_added": len(embeddings),
        "total_docs": vector_store.ntotal,
    }
