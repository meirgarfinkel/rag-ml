import logging
import re
from typing import List, Dict, Literal

from app.services.embedding_service import EmbeddingModel
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

MIN_CHUNK_LENGTH = 20
ChunkingStrategy = Literal["fixed", "sentence"]


# -------------------------------------------------
# Chunking strategies
# -------------------------------------------------


def fixed_chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Deterministic fixed-size overlapping chunker."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

        start += step

    return chunks


def sentence_aware_chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Sentence-aware chunking with hard size limits.
    Uses simple regex sentence splitting.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(current) + len(sentence) > chunk_size:
            if len(current) >= MIN_CHUNK_LENGTH:
                chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if len(current) >= MIN_CHUNK_LENGTH:
        chunks.append(current.strip())

    # Optional overlap (character-based)
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue

            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            combined = f"{overlap_text} {chunk}".strip()
            overlapped.append(combined[:chunk_size])

        chunks = overlapped

    return chunks


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkingStrategy = "sentence",
) -> List[str]:
    """Dispatch chunking strategy."""
    if strategy == "sentence":
        return sentence_aware_chunk_text(text, chunk_size, chunk_overlap)
    if strategy == "fixed":
        return fixed_chunk_text(text, chunk_size, chunk_overlap)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


# -------------------------------------------------
# Ingestion pipeline
# -------------------------------------------------


def process_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    chunking_strategy: ChunkingStrategy = "sentence",
) -> Dict[str, int]:
    """
    Chunk text, embed chunks, and store vectors.
    """

    if not text.strip():
        return {
            "chunks_added": 0,
            "total_docs": vector_store.ntotal,
        }

    logger.info(
        "Processing document '%s' (%d chars) using %s chunking",
        doc_id,
        len(text),
        chunking_strategy,
    )

    # Step 1: Chunk
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=chunking_strategy,
    )

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
            "chunk_size": len(chunk),
            "chunking_strategy": chunking_strategy,
        }
        for i, chunk in enumerate(chunks)
    ]

    # Step 4: Store
    vector_store.add_embeddings(embeddings, metadatas)

    return {
        "chunks_added": len(embeddings),
        "total_docs": vector_store.ntotal,
    }
