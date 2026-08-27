"""rag.py — coordinates retrieval and LLM answer generation.

Retrieves relevant source-code chunks for a natural-language question,
formats them as grounding context, and passes that context to DeepSeek
to generate an answer based on the indexed codebase.
"""

from llm import generate_answer
from retriever import retrieve_chunks


def build_context(retrieved_chunks: list[dict]) -> str:
    """Formats retrieved code chunks into context for the LLM.

    Args:
        retrieved_chunks: Chunks returned by retrieve_chunks(), 
            including source code and metadata.

    Returns:
        A formatted string containing the retrieved source code together
        with file and line information.
    """
    context_sections = []

    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]

        # Include source information so the LLM knows where each chunk came from.
        section = (
            f"Source: {metadata['file']}:"
            f"{metadata['start_line']}-{metadata['end_line']}\n"
            f"Name: {metadata['name']}\n\n"
            f"{chunk['text']}"
        )

        context_sections.append(section)

    # Separators between chunks.
    return "\n\n---\n\n".join(context_sections)


def answer_question(question: str, top_k: int = 5) -> dict:
    """Answers a question using retrieved code as grounding context.

    Args:
        question: Natural-language question about the indexed codebase.
        top_k: Number of relevant code chunks to retrieve.

    Returns:
        A dictionary containing the generated answer and metadata for the
        source chunks used as context.
    """
    # Retrieve the code chunks most semantically related to the question.
    retrieved_chunks = retrieve_chunks(question, top_k)

    # Convert the retrieved chunks into context that DeepSeek can read.
    context = build_context(retrieved_chunks)

    # Generate an answer using only the retrieved code as grounding context.
    answer = generate_answer(question, context)

    sources = [
        {
            "file": chunk["metadata"]["file"],
            "name": chunk["metadata"]["name"],
            "start_line": chunk["metadata"]["start_line"],
            "end_line": chunk["metadata"]["end_line"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    test_question = "How are command line arguments parsed?"

    result = answer_question(test_question, top_k=5)

    print(f"\nQUESTION:\n{test_question}")
    print(f"\nANSWER:\n{result['answer']}")

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"- {source['file']}:"
            f"{source['start_line']}-{source['end_line']} "
            f"({source['name']})"
        )