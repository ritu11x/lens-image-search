from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter
from typing import List
import torch
import numpy as np
import faiss
import json
import re
import os
import time
import io
from PIL import Image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

search_latencies = []
build_status = {"status": "idle", "processed": 0, "total": 0, "message": ""}


def safe_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return filename


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("thumbnails", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

print("BLIP model load ho raha hai...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model load ho gaya! ✅")

embeddings = np.load("embeddings.npy").astype('float32')
filenames = np.load("filenames.npy")
embedding_dim = embeddings.shape[1]

with open("keywords.json", "r") as f:
    keywords_data = json.load(f)

with open("captions.json", "r") as f:
    captions_data = json.load(f)

faiss.normalize_L2(embeddings)
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)
index.hnsw.efSearch = 64
print(f"Index ready! Total vectors: {index.ntotal}")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "and",
    "or", "but", "if", "then", "there", "here", "this", "that", "these",
    "those", "it", "its", "as", "some", "their", "his", "her"
}


def extract_query_keywords(query_text):
    words = re.findall(r'\b[a-z]+\b', query_text.lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)


def keyword_match_score(query_keywords, image_keywords):
    if not query_keywords:
        return 0.0
    image_keywords_set = set(image_keywords)
    overlap = query_keywords & image_keywords_set
    return len(overlap) / len(query_keywords)


@app.get("/search")
def search(
    query: str,
    top_k: int = 6,
    clip_weight: float = 0.8,
    keyword_weight: float = 0.2,
    min_score: float = 0.0,
    candidate_pool: int = 50
):
    start_time = time.time()
    templated_query = f"a photo of {query}"
    text_inputs = processor(text=[templated_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs)
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]
    query_embedding = query_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, candidate_pool)
    query_keywords = extract_query_keywords(query)

    scored_results = []
    for i, idx in enumerate(indices[0]):
        filename = str(filenames[idx])
        clip_distance = distances[0][i]
        clip_similarity = 1 - (clip_distance / 2)

        image_keywords = keywords_data.get(filename, [])
        kw_score = keyword_match_score(query_keywords, image_keywords)

        final_score = (clip_weight * clip_similarity) + (keyword_weight * kw_score)

        if final_score >= min_score:
            scored_results.append({
                "filename": filename,
                "caption": captions_data.get(filename, ""),
                "keywords": image_keywords[:5],
                "clip_score": round(float(clip_similarity), 4),
                "keyword_score": round(float(kw_score), 4),
                "final_score": round(float(final_score), 4)
            })

    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    elapsed_ms = (time.time() - start_time) * 1000
    search_latencies.append(elapsed_ms)
    if len(search_latencies) > 100:
        search_latencies.pop(0)

    return {
        "query": query,
        "results": scored_results[:top_k],
        "latency_ms": round(elapsed_ms, 2),
        "params": {
            "clip_weight": clip_weight,
            "keyword_weight": keyword_weight,
            "min_score": min_score
        }
    }


@app.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...), top_k: int = 6):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_image_features(**image_inputs)

    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]

    query_embedding = query_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        filename = str(filenames[idx])
        clip_distance = distances[0][i]
        clip_similarity = 1 - (clip_distance / 2)
        results.append({
            "filename": filename,
            "caption": captions_data.get(filename, ""),
            "similarity": round(float(clip_similarity), 4)
        })

    return {
        "uploaded_filename": file.filename,
        "results": results
    }


@app.get("/")
def home():
    return {"message": "Hybrid Image Search API is running. Visit /docs to explore the API."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "index_size": index.ntotal,
        "total_images": len(filenames)
    }


@app.get("/stats")
def stats():
    all_keywords = []
    for kw_list in keywords_data.values():
        all_keywords.extend(kw_list)

    top_keywords = Counter(all_keywords).most_common(10)
    avg_caption_words = sum(len(c.split()) for c in captions_data.values()) / len(captions_data) if captions_data else 0

    return {
        "total_images": len(filenames),
        "embedding_dimension": embedding_dim,
        "avg_caption_words": round(avg_caption_words, 1),
        "top_keywords": [{"word": w, "count": c} for w, c in top_keywords],
        "avg_search_latency_ms": round(sum(search_latencies) / len(search_latencies), 2) if search_latencies else None
    }


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    global embeddings, filenames, index

    uploaded = []
    skipped = []

    for file in files:
        image_bytes = await file.read()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            skipped.append(file.filename)
            continue

        safe_name = safe_filename(file.filename)
        save_path = os.path.join("images", safe_name)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        thumb = image.copy()
        thumb.thumbnail((200, 200))
        thumb.save(os.path.join("thumbnails", safe_name))

        image_inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            new_embedding = model.get_image_features(**image_inputs)
        if not isinstance(new_embedding, torch.Tensor):
            new_embedding = new_embedding.pooler_output if hasattr(new_embedding, 'pooler_output') else new_embedding[0]
        new_embedding = new_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
        faiss.normalize_L2(new_embedding)

        blip_inputs = blip_processor(image, return_tensors="pt")
        blip_output = blip_model.generate(**blip_inputs, max_new_tokens=30)
        caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)
        kws = list(extract_query_keywords(caption))

        index.add(new_embedding)
        embeddings = np.vstack([embeddings, new_embedding])
        filenames = np.append(filenames, safe_name)
        captions_data[safe_name] = caption
        keywords_data[safe_name] = kws

        uploaded.append({"filename": safe_name, "caption": caption, "keywords": kws})

    np.save("embeddings.npy", embeddings)
    np.save("filenames.npy", filenames)
    with open("captions.json", "w") as f:
        json.dump(captions_data, f, indent=2)
    with open("keywords.json", "w") as f:
        json.dump(keywords_data, f, indent=2)

    return {"uploaded": uploaded, "skipped": skipped, "total_images_now": index.ntotal}


def run_build_index_task():
    global embeddings, filenames, index, build_status

    all_files_on_disk = set(sorted(os.listdir("images")))
    already_indexed = set(filenames.tolist())
    new_files = sorted(all_files_on_disk - already_indexed)

    if not new_files:
        build_status = {"status": "done", "processed": 0, "total": 0, "message": "No new images found — all images are already indexed."}
        return

    build_status = {"status": "running", "processed": 0, "total": len(new_files), "message": "Indexing in progress..."}

    new_embeddings = []
    valid_files = []
    skipped = []

    for i, fname in enumerate(new_files):
        path = os.path.join("images", fname)
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            skipped.append(fname)
            build_status["processed"] = i + 1
            continue

        thumb = image.copy()
        thumb.thumbnail((200, 200))
        thumb.save(os.path.join("thumbnails", fname))

        image_inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**image_inputs)
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output if hasattr(emb, 'pooler_output') else emb[0]
        emb = emb.squeeze().numpy().astype('float32')
        new_embeddings.append(emb)
        valid_files.append(fname)

        blip_inputs = blip_processor(image, return_tensors="pt")
        blip_output = blip_model.generate(**blip_inputs, max_new_tokens=30)
        caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)
        captions_data[fname] = caption
        keywords_data[fname] = list(extract_query_keywords(caption))

        build_status["processed"] = i + 1

    if valid_files:
        new_embeddings = np.array(new_embeddings).astype('float32')
        faiss.normalize_L2(new_embeddings)
        index.add(new_embeddings)
        embeddings = np.vstack([embeddings, new_embeddings])
        filenames = np.append(filenames, valid_files)

        np.save("embeddings.npy", embeddings)
        np.save("filenames.npy", filenames)
        with open("captions.json", "w") as f:
            json.dump(captions_data, f, indent=2)
        with open("keywords.json", "w") as f:
            json.dump(keywords_data, f, indent=2)

    skip_note = f" Skipped: {', '.join(skipped)}." if skipped else ""
    build_status = {
        "status": "done",
        "processed": len(valid_files),
        "total": len(new_files),
        "message": f"Successfully indexed {len(valid_files)} new image(s).{skip_note}"
    }


@app.post("/build-index")
def build_index(background_tasks: BackgroundTasks):
    if build_status["status"] == "running":
        return {"message": "An indexing build is already in progress.", "status": "running"}

    background_tasks.add_task(run_build_index_task)
    return {"message": "Index build started in the background.", "status": "started"}


@app.get("/build-status")
def get_build_status():
    return build_status