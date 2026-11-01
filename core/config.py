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
INDEX_JSON_FILE = BASE_DIR / "index_with_keywords.json"

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

# Base curated sample images that belong to the core demonstration set
BASE_SAMPLE_IMAGES = {
    "breakfast_table.jpg", "city_night.jpg", "coffee_food_table.jpg",
    "flower_macro.jpg", "golden_retriever_dog.jpg", "mountain_lake.jpg",
    "person_by_water.jpg", "person_smiling.jpg", "red_sports_car.jpg",
    "red_sports_car_track.jpg", "tech_laptop.jpg", "blue_sports_car.jpg",
    "white_sports_car.jpg"
}

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

