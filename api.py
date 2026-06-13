from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat_with_memory import RAGChatbot
from contextlib import asynccontextmanager
import uuid

sessions: dict[str, RAGChatbot] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RAG API server starting...")
    yield
    sessions.clear()
    print("RAG API Server shutdown.")

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the RAG chatbot."""
    session_id = request.session_id or str(uuid.uuid4())
    if not session_id in sessions:
        sessions[session_id] = RAGChatbot()

    try:
        answer = sessions[session_id].ask(request.question)
        return ChatResponse(answer=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents():
    """Trigger document re-ingestion."""
    from ingest import load_documents, split_documents, create_vector_store
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
    return {"status": "ok", "chunks": len(chunks)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

