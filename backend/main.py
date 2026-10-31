import os
import shutil
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from rag import (
    process_document, ask_question, stream_ask_question,
    clear_all_documents, get_system_status, get_project_name,
    get_model_type, UPLOADS_PATH, remove_document, list_documents,
    warmup_models
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads models/embeddings and runs warmup once at startup."""
    try:
        warmup_models()
    except Exception as e:
        print(f"[Startup Warning] Model warmup notice: {e}")
    yield


app = FastAPI(title="Project Documentation SLM Chatbot", version="1.0.0", lifespan=lifespan)

# Allow browser to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    stream: bool = False
    history: list[dict] | None = None
    mode: str = "auto"
    selected_doc: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list
    mode: str
    escalate: bool = False
    project_name: str | None = None
    model_type: str | None = None

@app.get("/health")
def health_check():
    """Simple endpoint to verify the server is running."""
    return {"status": "ok", "message": "RAG Chatbot is running!"}

@app.get("/status")
def system_status():
    """Live system and Ollama model status."""
    return get_system_status()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a PDF or TXT file, validate it, process it, and index it.
    - Enforces 20 MB file size limit.
    - Detects scanned / image-only PDFs and reports clearly.
    - Returns chunk count on success.
    """
    # Validate file type
    allowed_types = [".pdf", ".txt"]
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Please upload a PDF or TXT file."
        )

    # Read full content to enforce size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is 20 MB."
        )

    # Sanitize filename and save
    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(UPLOADS_PATH, safe_name)
    os.makedirs(UPLOADS_PATH, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        chunks = process_document(file_path)
        return {
            "message": f"Successfully processed '{safe_name}'",
            "chunks_created": chunks,
            "filename": safe_name,
            "status": "ready"
        }
    except ValueError as e:
        # Covers scanned PDF / empty text / unsupported format errors
        # Remove the saved file since indexing failed
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/documents")
def list_uploaded_documents():
    """List all currently indexed documents with their chunk counts."""
    return {"documents": list_documents()}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Remove a single document's chunks from the index and delete the file from disk.
    Does not clear other documents.
    """
    try:
        result = remove_document(filename)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Main chat endpoint. Supports both progressive streaming (SSE) and full JSON response.
    Performs RAG if documents exist, direct conversational LLM otherwise.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.stream:
        return StreamingResponse(
            stream_ask_question(
                request.question,
                history=request.history,
                mode=request.mode,
                selected_doc=request.selected_doc
            ),
            media_type="text/event-stream"
        )
    
    try:
        result = ask_question(
            request.question,
            history=request.history,
            mode=request.mode,
            selected_doc=request.selected_doc
        )
        result["project_name"] = result.get("project_name", get_project_name())
        result["model_type"] = result.get("model_type", get_model_type())
        result["escalate"] = result.get("escalate", False)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/clear")
def clear_documents():
    """Remove all uploaded document embeddings."""
    return clear_all_documents()

# Serve frontend files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))
