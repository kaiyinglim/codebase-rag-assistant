from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class IndexRequest(BaseModel):
    repo_path: str


class IndexResponse(BaseModel):
    status: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_repo(request: IndexRequest):
    # TODO (Day 2-3): actually chunk + embed the repo
    return IndexResponse(status="stubbed", chunks_indexed=0)


@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest):
    # TODO (Day 4): actually run retrieval + LLM
    return QueryResponse(
        answer=f"You asked: {request.question} (not implemented yet)",
        sources=[],
    )