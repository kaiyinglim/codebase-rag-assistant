from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from chunker import chunk_repo
from embedder import embed_and_store_chunks
from rag import answer_question
from dependency_analyzer import find_dependents

app = FastAPI()


# --- Request/response schemas -----------------------------------------
# These define the exact shape of data each endpoint accepts and returns.
# FastAPI uses them to automatically validate incoming requests and to
# generate the interactive docs at /docs.

class IndexRequest(BaseModel):
    """Input for POST /index — the repo to chunk and embed."""
    repo_path: str  # filesystem path to the target repo, e.g. "../httpie"


class IndexResponse(BaseModel):
    """Output of POST /index — confirms indexing completed."""
    status: str
    chunks_indexed: int  # number of function/class chunks stored


class QueryRequest(BaseModel):
    """Input for POST /query — a natural language question about the repo.

    Attributes:
        question: Natural-language question about the indexed repository.
        top_k: Number of code chunks to retrieve for the question.
    """

    question: str
    top_k: int = Field(
        default=5, 
        ge=1,   # greater than or equal to 1
        le=20   # less than or equal to 20
        )  
    
    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        """Checks that the question contains meaningful text.

        Args:
            question: Question supplied in the API request.

        Returns:
            The question with surrounding whitespace removed.

        Raises:
            ValueError: If the question is empty or only contains whitespace.
        """
        question = question.strip()

        # Do not run retrieval when there is no actual question.
        if not question:
            raise ValueError("Question cannot be empty.")

        return question


class QueryResponse(BaseModel):
    """Output of POST /query — the answer plus the chunks used to produce it."""
    answer: str
    sources: list[dict]  # e.g. [{"file": ..., "start_line": ..., "end_line": ...}]


class DependencyResponse(BaseModel):
    """Output for GET /dependencies.

    Attributes:
        dependency: Module or imported symbol that was searched for.
        dependents: Repository-relative files that depend on the target.
    """

    dependency: str
    dependents: list[str]


# --- Endpoints -----------------------------------------------------------

@app.get("/health")
def health():
    """
    Function Description:
        Basic liveness check — confirms the service is running.
    """
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_repo(request: IndexRequest):
    """Indexes a Python repository for semantic code retrieval.

    Parses the repository into function/class-level chunks, generates
    embeddings for those chunks, and stores them in the persistent
    ChromaDB collection.

    Args:
        request: Request containing the filesystem path of the repository
            to index.

    Returns:
        An IndexResponse reporting how many chunks were indexed.
    """
    # Parse the target repository into source-code chunks.
    chunks = chunk_repo(request.repo_path)

    # Generate local Jina embeddings and persist them in ChromaDB.
    stored_count = embed_and_store_chunks(chunks)

    return IndexResponse(
        status="success",
        chunks_indexed=stored_count,
    )


@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest):
    """Answers a question about the indexed codebase.

    Retrieves the most relevant source-code chunks from ChromaDB and
    provides them to DeepSeek as grounding context for the answer.

    Args:
        request: Request containing the question and number of chunks
            to retrieve.

    Returns:
        A QueryResponse containing the generated answer and source
        metadata for the retrieved chunks.
    """
    # Run retrieval and grounded LLM generation through the RAG pipeline.
    result = answer_question(
        question=request.question,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@app.get("/dependencies", response_model=DependencyResponse)
def get_dependencies(repo_path: str, dependency: str):
    """Finds repository files that depend on a module or imported symbol.

    Uses deterministic AST-based import analysis rather than semantic
    retrieval, because dependency relationships can be identified directly
    from Python import statements.

    Args:
        repo_path: Filesystem path to the repository to analyze.
        dependency: Module or imported symbol to search for.

    Returns:
        A DependencyResponse containing the requested dependency and all
        repository files that import it.
    """
    try:
        # Find all files in the repository that import this dependency.
        dependents = find_dependents(
            repo_path=repo_path,
            dependency=dependency,
        )

    except ValueError as error:
        # Return a 400 response when the input is invalid.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    # Return the dependency together with the matching files.
    return DependencyResponse(
        dependency=dependency,
        dependents=dependents,
    )