"""retriever.py — retrieves semantically relevant code chunks from ChromaDB.

Embeds a natural-language question using the same local embedding model used
during indexing, then searches the ChromaDB collection for the closest stored
code embeddings.
"""

from embedder import COLLECTION_NAME, embed_texts, get_chroma_client


def retrieve_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Retrieves the most semantically relevant code chunks for a question.

    Args:
        question: Natural-language question about the indexed codebase.
        top_k: Maximum number of matching chunks to return.

    Returns:
        A list of dictionaries containing the retrieved source code,
        metadata, and vector distance for each matching chunk.

    Raises:
        ValueError: If the question is empty, top_k is invalid, 
            or no code has been indexed yet.
    """
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Embed the question using the same model used for the code chunks.
    query_embedding = embed_texts([question])[0]

    chroma_client = get_chroma_client()

    # Get the collection, or create an empty one if indexing has not happened yet.
    collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME
    )

    # Stop early if there is no indexed code available to search.
    indexed_count = collection.count()

    if indexed_count == 0:
        raise ValueError(
            "No indexed code found. Run /index first."
        )

    # Never ask Chroma for more results than are actually stored.
    result_count = min(top_k, indexed_count)

    # Search for the stored code vectors closest to the question vector.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Combine each retrieved code chunk with its metadata and distance.
    return [
        {
            "text": document,       # actual source code
            "metadata": metadata,   # file, function name, line numbers
            "distance": distance,   # how far the query vector is from that chunk vector
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]


if __name__ == "__main__":
    test_questions = [
        "how are command line arguments parsed?",
        "how is authentication handled?",
        "how are HTTP requests created?",
    ]

    for question in test_questions:
        print(f"\nQUESTION: {question}")

        for result in retrieve_chunks(question, top_k=3):
            metadata = result["metadata"]

            print(
                f"\n--- {metadata['name']} "
                f"({metadata['file']}:{metadata['start_line']}-"
                f"{metadata['end_line']}) ---"
            )
            print(f"Distance: {result['distance']:.4f}")
            print(result["text"][:300])