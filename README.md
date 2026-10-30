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
core/
├── config.py           Centralized configuration, paths, constants, device selection
├── models.py           Singleton model loader for CLIP and BLIP with device abstraction
├── search_engine.py    FAISS HNSW index management, BM25 scoring, synonym expansion, find_similar
└── indexer.py          Batched CLIP embedding, BLIP captioning, thumbnail generator, atomic persistence

app.py                  FastAPI backend with non-blocking threadpool delegation (main entry point)
index.html              Redesigned modern glassmorphism frontend (served directly at /)
requirements.txt        Pinned dependencies for clean setup

images/                 Image dataset (curated sample images included)
thumbnails/             Generated high-resolution thumbnails
embeddings.npy          Normalized CLIP embeddings (512-dim)
filenames.npy           Filenames aligned with embeddings
captions.json           BLIP captions per image
keywords.json           Extracted keywords per image

build_index.py          One-time batch indexing script
generate_captions.py    BLIP caption generation script
extract_keywords.py     Keyword extraction script
hybrid_search.py        Standalone hybrid search logic
auto_eval.py            Automated precision evaluation
compute_metrics.py      Precision/recall calculation
eval_set.json           Evaluation queries and candidates
```

## Running the Application

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```

3. **Open in browser:**
   Open `http://127.0.0.1:8000` in any web browser to access the full search engine.

## New Features & Interview Highlights

### 1. 🌟 Composed Multimodal Retrieval (Image + Text Refinement)
Goes beyond standard single-vector retrieval by performing vector arithmetic directly in the CLIP latent space:
$$\vec{v}_{\text{composed}} = \text{normalize}\left(\alpha \cdot \vec{v}_{\text{img}} + (1 - \alpha) \cdot \vec{v}_{\text{modifier}}\right)$$
- **Use Case**: An image of a red car can be modified by typing `"on race track"` or `"in rain"`, immediately retrieving the racing version of the car.
- **Available via**:
  - Image search preview with the `✨ Refine` input box
  - Lightbox modal with one-click composed refinement search
  - Dedicated endpoint: `POST /search-composed`

### 2. 🚫 Negative Prompting & Vector Subtraction (Search with Exclusion)
Allows users to exclude specific colors, objects, or concepts from visual and semantic search:
$$\vec{v}_{\text{query}} = \text{normalize}\left(\vec{v}_{\text{pos}} - \lambda \cdot \vec{v}_{\text{neg}}\right)$$
- Accompanied by negative keyword penalties to eliminate unwanted concepts.
- **User Interface**: Dedicated `- Exclude:` input bar with real-time tag badge and inline syntax support (`cars --without red`, `dog --not grass`).
- **Endpoint**: `GET /search?query=car&negative_query=red`

### 3. 🎯 Smart Relevance Guard (Zero Random / Irrelevant Images)
- Standard nearest-neighbor searches always return $K$ images, even when the query has zero semantic connection (returning noisy nearest neighbors with scores around $0.18 - 0.22$).
- Lens implements a **Smart Relevance Guard**: candidates with zero BM25 keyword overlap must achieve a strict cosine similarity threshold ($\ge 0.258$). Queries with no relevant match cleanly display an explanation card with curated explore suggestion chips instead of deceptive random photos.

### 4. 🎛️ Minimalist Stepper Count Control
- Replaced clunky dropdown menus with an aesthetic `−` / `+` stepper counter directly inside the search bar, with instant keyboard and click feedback.

### 5. ⚡ Architecture & Performance Improvements
- **Instant "Find Similar" Discovery**: Click the Lens icon on any image card to immediately retrieve visually similar photos without re-uploading.
- **BM25 Reranking & Synonym Expansion**: Replaces naive keyword set-overlap with Okapi BM25 and concept synonym expansion (`person` $\to$ `man`, `woman`, `people`), eliminating false zero scores.
- **Interactive Lightbox Modal**: View full-size images, complete BLIP captions, keyword chips, and granular CLIP vs BM25 score meters.
- **Non-Blocking Architecture**: Synchronous PyTorch and FAISS operations run on threadpools via `run_in_threadpool`, ensuring FastAPI's asyncio event loop is never blocked.
- **Atomic File Persistence**: Saves indexes and metadata via atomic temporary file renaming to prevent file corruption during writes.
- **Device-Aware Hardware Detection**: Auto-detects and utilizes CUDA/GPU acceleration when available, with graceful CPU fallback.

## License

MIT

