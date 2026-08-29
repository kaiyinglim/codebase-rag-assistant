"""embedder.py — embeds chunks and stores them in a local ChromaDB collection.

Takes the output of chunker.py (a list of chunk dicts) and embeds them in
batches using a local SentenceTransformer model, then upserts the vectors
alongside each chunk's text and metadata into ChromaDB for semantic search.
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "codebase_chunks"
BATCH_SIZE = 8

# Load the code-focused embedding model once when this module starts.
model = SentenceTransformer(
    EMBEDDING_MODEL,
    trust_remote_code=True,
)


def get_chroma_client():
    """Returns a Chroma client that persists data to disk.

    A persistent client means the vector store survives between runs, so
    /index doesn't need to recreate the database whenever the server
    restarts.
    """
    return chromadb.PersistentClient(path=CHROMA_PATH)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds multiple texts locally using the SentenceTransformer model.

    Args:
        texts: A batch of text strings to embed.

    Returns:
        A list of embedding vectors, in the same order as `texts`.
    """
    embeddings = model.encode(texts)
    return embeddings.tolist()


def reset_collection():
    """Deletes the existing collection so a new index starts empty.

    Without this, indexing another repository would upsert into the same
    collection and leave old chunks searchable.
    """
    chroma_client = get_chroma_client()

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except ValueError:
        # No existing collection means there is nothing to clear.
        pass

    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def embed_and_store_chunks(chunks: list[dict]) -> int:
    """Embeds chunks in batches and stores them in a fresh ChromaDB collection.

    Clears any previously indexed chunks first so each /index call represents
    exactly one repository. Then processes chunks BATCH_SIZE at a time and
    upserts each embedding alongside its source code and metadata.

    Args:
        chunks: A list of chunk dicts, as produced by chunker.chunk_repo.
            Each must have "text", "file", "name", "type", "start_line",
            and "end_line" keys.

    Returns:
        The number of chunks successfully embedded and stored.
    """
    collection = reset_collection()

    stored_count = 0

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        embeddings = embed_texts(texts)

        # Build deterministic IDs from each chunk's source information.
        ids = [
            f"{chunk['file']}::{chunk['type']}::{chunk['name']}::{chunk['start_line']}"
            for chunk in batch
        ]

        metadatas = [
            {
                "file": chunk["file"],
                "name": chunk["name"],
                "type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }
            for chunk in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        stored_count += len(batch)
        print(f"  Embedded {stored_count}/{len(chunks)} chunks...")

    return stored_count


if __name__ == "__main__":
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    count = embed_and_store_chunks(chunks)
    print(f"Done. Stored {count} chunks in ChromaDB at {CHROMA_PATH}.")