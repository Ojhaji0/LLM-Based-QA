"""
API endpoints for the QA system using FastAPI.
Supports synchronous and streaming responses, and PDF upload/ingestion.
"""
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from loguru import logger
import time
import asyncio
import os
import json
import shutil

# Import RAG Pipeline components
from src.pipeline.rag_chain import RAGPipeline
from src.generation.prompt_builder import build_prompt
from src.generation.citation_validator import validate_citations
from src.ingestion.ingest_pipeline import IngestionPipeline

app = FastAPI(title="University RAG QA API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the 'frontend' directory
FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")
RAW_PDFS_DIR = os.path.join(os.getcwd(), "data", "raw_pdfs")
os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(RAW_PDFS_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Initialize pipeline lazily
rag_pipeline = None
is_indexing = False

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    logger.info("Starting up FastAPI server and initializing RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized.")


class QueryRequest(BaseModel):
    question: str
    stream: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm": "phi3:latest",
        "retrieval": "hybrid-bm25-dense-rerank"
    }


@app.get("/documents")
def list_documents():
    """List all documents that have been uploaded."""
    files = [f for f in os.listdir(RAW_PDFS_DIR) if f.endswith((".pdf", ".docx", ".doc"))]
    return {"documents": files, "count": len(files)}


@app.get("/status")
def get_status():
    """Return the current indexing status."""
    global is_indexing
    return {"indexing": is_indexing}


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a document file and trigger re-ingestion in the background.
    """
    if not file.filename.lower().endswith((".pdf", ".docx", ".doc")):
        return {"error": "Only PDF and DOCX files are supported."}

    save_path = os.path.join(RAW_PDFS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"PDF uploaded: {file.filename}. Starting background ingestion...")

    # Run ingestion in background so API doesn't block
    background_tasks.add_task(reingest_and_reload, save_path, file.filename)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "message": "PDF uploaded successfully! Indexing in background (~30s). You can ask questions after indexing is done."
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str, background_tasks: BackgroundTasks):
    """Delete a PDF and re-ingest remaining."""
    file_path = os.path.join(RAW_PDFS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted PDF: {filename}. Starting background ingestion...")
        background_tasks.add_task(reingest_all_and_reload)
        return {"status": "deleted", "message": f"{filename} deleted."}
    return {"error": "File not found"}

def reingest_all_and_reload():
    """Re-run ingestion for all existing PDFs after a deletion."""
    global rag_pipeline, is_indexing
    if is_indexing:
        logger.warning("Indexing already in progress. Skipping duplicate request.")
        return
        
    is_indexing = True
    try:
        logger.info("Background full re-ingestion started...")
        # Clear vectorstore directory completely to avoid ghost chunks
        vs_dir = os.path.join(os.getcwd(), "vectorstore")
        if os.path.exists(vs_dir):
            shutil.rmtree(vs_dir)
            
        pipeline = IngestionPipeline(
            raw_pdfs_dir=RAW_PDFS_DIR,
            vectorstore_dir=vs_dir
        )
        pipeline.process_all()
        logger.info("Re-ingestion complete. Reloading retriever...")

        # Reload the BM25 index in the live RAG pipeline
        if rag_pipeline:
            from src.retrieval.bm25_retriever import BM25Retriever
            # Recreate vectorstore connection
            from src.retrieval.vector_store import get_vector_store
            rag_pipeline.retriever.vectorstore = get_vector_store(vs_dir)
            rag_pipeline.retriever.bm25 = BM25Retriever(rag_pipeline.retriever.vectorstore)
            logger.info("Pipeline reloaded successfully!")
    except Exception as e:
        logger.error(f"Background re-ingestion failed: {e}")
    finally:
        is_indexing = False


def reingest_and_reload(pdf_path: str, filename: str):
    """Re-run the ingestion pipeline and reload the BM25 index."""
    global rag_pipeline, is_indexing
    if is_indexing:
        logger.warning("Indexing already in progress. Skipping duplicate request.")
        return
        
    is_indexing = True
    try:
        logger.info(f"Background ingestion started for: {filename}")
        pipeline = IngestionPipeline(
            raw_pdfs_dir=RAW_PDFS_DIR,
            vectorstore_dir=os.path.join(os.getcwd(), "vectorstore")
        )
        pipeline.process_all()
        logger.info("Ingestion complete. Reloading retriever BM25 index...")

        # Reload the BM25 index in the live RAG pipeline
        if rag_pipeline:
            from src.retrieval.bm25_retriever import BM25Retriever
            from src.retrieval.vector_store import get_vector_store
            vs_dir = os.path.join(os.getcwd(), "vectorstore")
            rag_pipeline.retriever.vectorstore = get_vector_store(vs_dir)
            rag_pipeline.retriever.bm25 = BM25Retriever(
                rag_pipeline.retriever.vectorstore
            )
            logger.info("BM25 index reloaded successfully. New PDF is ready to query!")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")
    finally:
        is_indexing = False


@app.post("/query")
async def query(req: QueryRequest):
    pipeline = rag_pipeline
    if not pipeline:
        return {"error": "RAG pipeline not initialized yet"}

    if req.stream:
        return await query_stream(req)

    logger.info(f"Sync query received: {req.question}")
    response = pipeline.ask(req.question)
    return response


async def query_stream(req: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    """
    logger.info(f"Stream query received: {req.question}")

    async def event_generator():
        pipeline = rag_pipeline
        if not pipeline:
            yield {"event": "error", "data": "Pipeline not initialized"}
            return

        t0 = time.time()

        # 1. Retrieve
        retrieved_chunks = pipeline.retriever.retrieve(req.question, top_k=5)

        # 2. Build Prompt
        prompt = build_prompt(req.question, retrieved_chunks)

        # Send metadata
        source_files = list(set([
            os.path.basename(chunk[1].get('source_file', ''))
            for chunk in retrieved_chunks
        ]))
        meta = {
            "retrieved_docs": len(retrieved_chunks),
            "sources": source_files
        }
        yield {"event": "metadata", "data": json.dumps(meta)}

        # 3. Stream LLM output
        full_answer = ""
        for chunk in pipeline.llm.stream(prompt):
            full_answer += chunk
            yield {"event": "token", "data": json.dumps({"text": chunk})}
            await asyncio.sleep(0.01)

        # 4. Post-generation citation validation
        validation = validate_citations(full_answer, source_files)

        latency = time.time() - t0
        final_meta = {
            "latency": latency,
            "citations_valid": validation["valid"],
            "citations_invalid": validation["invalid"],
            "hallucinated_citations": validation["hallucinated_citations"]
        }
        yield {"event": "done", "data": json.dumps(final_meta)}

    return EventSourceResponse(event_generator(), ping=15)
