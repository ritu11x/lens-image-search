import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "images"
THUMBNAILS_DIR = BASE_DIR / "thumbnails"

EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"
FILENAMES_FILE = BASE_DIR / "filenames.npy"
CAPTIONS_FILE = BASE_DIR / "captions.json"
KEYWORDS_FILE = BASE_DIR / "keywords.json"

# Auto-detect best available device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

EMBEDDING_DIM = 512
HNSW_M = 32
HNSW_EF_SEARCH = 64
HNSW_EF_CONSTRUCTION = 64

DEFAULT_TOP_K = 6
DEFAULT_CLIP_WEIGHT = 0.8
DEFAULT_KEYWORD_WEIGHT = 0.2
DEFAULT_CANDIDATE_POOL = 50

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
