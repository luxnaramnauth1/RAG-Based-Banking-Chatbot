"""
Orange RAG Banking Assistant — Backend API

A REST API wrapping the retrieval-augmented generation pipeline. Run with:

    uvicorn api:app --reload --port 8000

Then try:
    curl http://localhost:8000/health
    curl http://localhost:8000/categories
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \\
         -d '{"question": "How much is the overdraft fee?"}'

Interactive API docs are auto-generated at http://localhost:8000/docs

Note: this uses the TF-IDF retrieval engine (retrieval_engine.py) so it runs
instantly with zero model downloads. To use real sentence-transformer
embeddings + an open-source LLM instead (the full production pipeline),
see ingest.py / chatbot.py — the /ask endpoint below is designed to be a
drop-in swap for that pipeline without changing the API contract.
"""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from retrieval_engine import RetrievalEngine

app = FastAPI(
    title="Orange — RAG Banking Assistant API",
    description="Retrieval-augmented Q&A over a banking policy knowledge base.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RetrievalEngine()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How much is the overdraft fee?"])
    top_k: int = Field(3, ge=1, le=10)


class SourceHit(BaseModel):
    id: str
    category: str
    title: str
    score: float


class AskResponse(BaseModel):
    ok: bool
    answer: str
    cited_source: Optional[str]
    sources: list[SourceHit]
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": len(engine.docs),
        "categories": len(engine.categories()),
    }


@app.get("/categories")
def categories():
    return engine.categories()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    t0 = time.perf_counter()
    result = engine.ask(question, k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    return AskResponse(
        ok=result["ok"],
        answer=result["answer"],
        cited_source=result["cited_source"],
        sources=[
            SourceHit(id=s["id"], category=s["category"], title=s["title"], score=round(s["score"], 4))
            for s in result["sources"]
        ],
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
