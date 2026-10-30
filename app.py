import os
import io
import time
import shutil
from typing import List, Optional
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool

from core.config import IMAGES_DIR, THUMBNAILS_DIR, DEVICE, EMBEDDING_DIM
from core.models import model_manager
from core.search_engine import search_engine
from core.indexer import indexer, sanitize_filename

search_latencies = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure directories exist and search engine loads existing index
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    search_engine.load_index()
    yield
    # Shutdown clean up if needed


app = FastAPI(
    title="Lens — Hybrid Semantic Image Search",
    description="Multimodal image search engine combining CLIP, BLIP, and FAISS HNSW.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS_DIR)), name="thumbnails")


from fastapi.responses import JSONResponse, FileResponse

@app.get("/")
def home():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "name": "Lens Image Search API",
        "version": "2.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "indexed_images": search_engine.index.ntotal if search_engine.index else 0
    }


@app.get("/api")
def api_info():
    return {
        "name": "Lens Image Search API",
        "version": "2.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "indexed_images": search_engine.index.ntotal if search_engine.index else 0
    }


@app.get("/health")
def health():
    index_size = search_engine.index.ntotal if search_engine.index else 0
    return {
        "status": "ok",
        "device": DEVICE,
        "index_ready": search_engine.index is not None,
        "index_size": index_size,
        "total_images": len(search_engine.filenames),
        "total_captions": len(search_engine.captions),
        "total_keywords": len(search_engine.keywords)
    }


@app.get("/stats")
def stats():
    all_kws = []
    for kw_list in search_engine.keywords.values():
        all_kws.extend(kw_list)

    from collections import Counter
    top_keywords = Counter(all_kws).most_common(12)
    captions = search_engine.captions
    avg_len = sum(len(c.split()) for c in captions.values()) / len(captions) if captions else 0.0

    return {
        "total_images": len(search_engine.filenames),
        "embedding_dimension": EMBEDDING_DIM,
        "device": DEVICE,
        "avg_caption_words": round(avg_len, 1),
        "top_keywords": [{"word": w, "count": c} for w, c in top_keywords],
        "avg_search_latency_ms": round(sum(search_latencies) / len(search_latencies), 2) if search_latencies else None,
        "total_searches": len(search_latencies)
    }


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="Text query to search for"),
    top_k: int = Query(6, ge=1, le=50),
    clip_weight: float = Query(0.8, ge=0.0, le=1.0),
    keyword_weight: float = Query(0.2, ge=0.0, le=1.0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    candidate_pool: int = Query(50, ge=5, le=200),
    negative_query: Optional[str] = Query(None, description="Concepts/keywords to subtract or exclude"),
    negative_weight: float = Query(0.45, ge=0.0, le=1.0)
):
    if search_engine.index is None or search_engine.index.ntotal == 0:
        return {
            "query": query,
            "results": [],
            "latency_ms": 0.0,
            "message": "Index is currently empty. Please upload images or run 'Build Index'."
        }

    start = time.time()
    results = search_engine.search_text(
        query=query,
        top_k=top_k,
        clip_weight=clip_weight,
        keyword_weight=keyword_weight,
        min_score=min_score,
        candidate_pool=candidate_pool,
        negative_query=negative_query,
        negative_weight=negative_weight
    )
    elapsed_ms = round((time.time() - start) * 1000, 2)
    search_latencies.append(elapsed_ms)
    if len(search_latencies) > 200:
        search_latencies.pop(0)

    return {
        "query": query,
        "negative_query": negative_query,
        "results": results,
        "latency_ms": elapsed_ms,
        "params": {
            "clip_weight": clip_weight,
            "keyword_weight": keyword_weight,
            "min_score": min_score,
            "negative_weight": negative_weight
        }
    }


@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(6, ge=1, le=50)
):
    if search_engine.index is None or search_engine.index.ntotal == 0:
        return {
            "uploaded_filename": file.filename,
            "results": [],
            "latency_ms": 0.0,
            "message": "Index is currently empty. Please upload images or run 'Build Index'."
        }

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    start = time.time()
    results = await run_in_threadpool(search_engine.search_image, image, top_k)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "uploaded_filename": file.filename,
        "results": results,
        "latency_ms": elapsed_ms
    }


@app.post("/search-composed")
async def search_composed(
    file: Optional[UploadFile] = File(None),
    base_filename: Optional[str] = Form(None),
    text_modifier: str = Form(""),
    image_weight: float = Form(0.5),
    top_k: int = Form(6)
):
    """Feature 1: Composed Multimodal Retrieval (Image + Text Refinement)."""
    if search_engine.index is None or search_engine.index.ntotal == 0:
        return {
            "results": [],
            "latency_ms": 0.0,
            "message": "Index is currently empty."
        }

    image = None
    if file is not None and file.filename:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

    start = time.time()
    results = await run_in_threadpool(
        search_engine.search_composed,
        image=image,
        base_filename=base_filename,
        text_modifier=text_modifier,
        image_weight=image_weight,
        top_k=top_k
    )
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "base_filename": base_filename,
        "text_modifier": text_modifier,
        "image_weight": image_weight,
        "results": results,
        "latency_ms": elapsed_ms
    }


@app.get("/find-similar/{filename}")
async def find_similar(
    filename: str,
    top_k: int = Query(6, ge=1, le=50)
):
    """Find visually similar images to an existing indexed image."""
    if search_engine.index is None or search_engine.index.ntotal == 0:
        raise HTTPException(status_code=404, detail="Index is empty.")

    start = time.time()
    results = await run_in_threadpool(search_engine.find_similar, filename, top_k)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    return {
        "target_filename": filename,
        "results": results,
        "latency_ms": elapsed_ms
    }


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    saved_filenames = []
    skipped = []

    for f in files:
        safe_name = sanitize_filename(f.filename)
        dest_path = IMAGES_DIR / safe_name
        contents = await f.read()

        try:
            # Validate image
            test_img = Image.open(io.BytesIO(contents))
            test_img.verify()
            with open(dest_path, "wb") as buffer:
                buffer.write(contents)
            saved_filenames.append(safe_name)
        except Exception:
            skipped.append(f.filename)

    if saved_filenames:
        # Run indexing in threadpool
        res = await run_in_threadpool(indexer.index_files, saved_filenames)
        return {
            "uploaded": saved_filenames,
            "skipped": skipped,
            "total_images_now": search_engine.index.ntotal if search_engine.index else 0,
            "message": f"Successfully indexed {len(saved_filenames)} image(s)."
        }

    return {
        "uploaded": [],
        "skipped": skipped,
        "total_images_now": search_engine.index.ntotal if search_engine.index else 0,
        "message": "No valid images could be saved."
    }


@app.post("/build-index")
def trigger_build_index(background_tasks: BackgroundTasks):
    current = indexer.get_status()
    if current["status"] == "running":
        return {"message": "Indexing is already running in background.", "status": "running"}

    background_tasks.add_task(indexer.build_index_for_new_images)
    return {"message": "Index build started in background.", "status": "started"}


@app.get("/build-status")
def get_build_status():
    return indexer.get_status()


@app.post("/privacy/clear-uploads")
async def clear_user_uploads():
    """Privacy feature: permanently deletes any user uploaded photos and resets to base sample set."""
    res = await run_in_threadpool(indexer.reset_and_rebuild_base_index)
    return res


@app.get("/privacy/info")
def privacy_info():
    """Returns local-first and zero-cloud privacy architecture metadata."""
    return {
        "architecture": "100% Local-First & In-Memory",
        "external_api_calls": False,
        "external_data_storage": False,
        "query_image_storage": "None (Processed strictly in-memory via io.BytesIO and discarded)",
        "telemetry_and_tracking": False,
        "user_data_sovereignty": "Full local ownership with 1-click privacy scrub"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860 if os.environ.get("SPACE_ID") else 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)