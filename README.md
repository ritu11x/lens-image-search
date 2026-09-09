# Lens — Hybrid Semantic Image Search

A local, offline-first image search engine that combines **CLIP** visual embeddings, **BLIP**-generated captions, and **FAISS HNSW** indexing to search a personal image collection using natural language — or by uploading an image to find visually similar ones.

## Why this project

Most portfolio image-search projects stop at "call CLIP, compute cosine similarity." This one goes further:

- **Hybrid retrieval**: a FAISS-based semantic pass is reranked using a tunable blend of CLIP similarity and keyword overlap from auto-generated captions — a retrieve-then-rerank pattern used in real production search systems.
- **Measured, not assumed, performance**: FAISS HNSW indexing was benchmarked against brute-force search on 1,000 images — see [Benchmarks](#benchmarks) for real numbers, not estimates.
- **Bidirectional search**: text-to-image and image-to-image search share the same CLIP embedding space and the same FAISS index.
- **Honest evaluation**: an automated precision evaluation is included, along with a documented finding about where exact-keyword ground truth undercounts semantic matches — see [Evaluation](#evaluation).

## Features

- Natural language image search
- Reverse image search (search by uploading an image)
- Adjustable CLIP-weight / keyword-weight hybrid scoring
- Multi-image drag-and-drop upload with automatic captioning
- Background index building (non-blocking) with live progress polling — safe for large batches
- Thumbnail generation
- Dataset statistics and system health endpoints, with server-side latency tracking
- Light and dark themes
- Filename sanitization and corrupted-file handling
- Fully offline after the initial model download — no data leaves the machine

## Tech stack

| Component | Choice |
|---|---|
| Visual/text embeddings | CLIP (`openai/clip-vit-base-patch32`) |
| Automatic captioning | BLIP (`Salesforce/blip-image-captioning-base`) |
| Approximate nearest-neighbor search | FAISS (`IndexHNSWFlat`) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Inference | PyTorch (CPU) |

## Architecture

```
Images ──► CLIP ──► embeddings (embeddings.npy)
       └─► BLIP ──► captions (captions.json) ──► keywords (keywords.json)

Text query ──► CLIP text encoder ──► FAISS HNSW search (candidate pool)
                                          │
                                          ▼
                          rerank: (clip_weight × CLIP score)
                                + (keyword_weight × keyword overlap)
                                          │
                                          ▼
                                   ranked results

Image query ──► CLIP image encoder ──► same FAISS index ──► ranked results
```

## Benchmarks

Measured on a 1,000-image dataset (mixed Unsplash + Flickr8k), CPU-only:

| Method | Pure search time (avg) |
|---|---|
| Brute-force cosine similarity | ~7.7 ms |
| FAISS HNSW (efSearch=64) | ~0.29 ms |

**~27x speedup**, with **100% recall@3** against brute-force ground truth after tuning `efSearch` from its default of 16 up to 64 (the initial setting gave a larger speedup at the cost of missing some true matches — a real accuracy/speed trade-off, not a free win).

Query accuracy was further improved by applying CLIP's zero-shot prompt-templating technique (wrapping short queries as `"a photo of {query}"`), which measurably improved results for short, abstract queries.

## Evaluation

An automated evaluation script (`auto_eval.py`) scores retrieval using an exact-keyword-match criterion (no manual labeling) across 10 test queries:

- Average Precision@5: **0.28**
- Average Precision@10: **0.24**

Precision varied sharply by query type: concrete object queries ("flower", "dog") scored 90–100%, while abstract/category queries ("person", "technology") scored near 0%. Investigating this revealed the cause: BLIP captions use specific words ("man", "woman") rather than the query's category word ("person"), so an exact-keyword ground truth systematically undercounts correct semantic matches. This is itself a useful finding — it demonstrates concretely why CLIP's semantic search adds value beyond literal keyword matching, rather than just asserting it.

## Setup

**Requirements:** Python 3.10+, ~5GB free disk space (for model downloads), no GPU required.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers timm faiss-cpu pillow numpy "fastapi[standard]"
```

Place images in an `images/` folder, then build the index:

```bash
python build_index.py
python generate_captions.py
python extract_keywords.py
```

Start the server:

```bash
fastapi dev app.py
```

Open `index.html` in a browser (the backend runs at `http://127.0.0.1:8000`).

## Project structure

```
app.py                  FastAPI backend — search, upload, indexing, stats (main entry point)
build_index.py          One-time / batch CLIP embedding generation
generate_captions.py    BLIP caption generation for the dataset
extract_keywords.py     Keyword extraction from captions
hybrid_search.py        Standalone hybrid search logic (used during development)
auto_eval.py            Automated precision evaluation (generates the numbers below)
index.html              Frontend (search UI, upload, stats, about)

search.py               Early standalone brute-force search script (exploration)
search_faiss.py         Early standalone FAISS search script (exploration)
test_clip.py            Script used to verify CLIP model loading during development
test_blip.py            Script used to verify BLIP model loading during development
test_image_embedding.py Script used to verify image embedding generation during development
create_eval_set.py      Interactive helper used to build the manual portion of the eval set
compute_metrics.py      Precision/recall calculation from eval_set.json
eval_set.json           Saved evaluation queries and candidate/relevant results

images/                 Image dataset (not committed — see .gitignore)
thumbnails/             Generated thumbnails (not committed)
embeddings.npy          CLIP embeddings for indexed images (not committed)
filenames.npy           Filenames aligned with embeddings.npy (not committed)
captions.json           BLIP captions per image (not committed)
keywords.json           Extracted keywords per image (not committed)
```

The `search.py` / `search_faiss.py` / `test_*.py` scripts were used while building the system incrementally (verifying CLIP, then FAISS, then BLIP, one piece at a time) and are kept for reference — `app.py` is the actual served application and is the only file that needs to run.

## Known limitations

- Search runs on CPU; a GPU would reduce embedding/captioning latency further.
- The evaluation set is small (10 queries) and uses an automated keyword-based proxy rather than fully manual relevance judgments — see [Evaluation](#evaluation) for why this matters.
- Background indexing runs as a single in-process task; a production system would use a proper task queue (e.g. Celery) instead of FastAPI's `BackgroundTasks`.

## License

MIT (or update to match your preference before publishing).
