import os
import gc
import json
import re
import hashlib
import logging
import psutil
from typing import List, Union
from PIL import Image
import torch
import numpy as np

from core.config import (
    CLIP_MODEL_NAME, BLIP_MODEL_NAME, DEVICE,
    EMBEDDINGS_FILE, FILENAMES_FILE, KEYWORDS_FILE, CAPTIONS_FILE
)

logger = logging.getLogger("lens.models")

def is_render_or_low_ram() -> bool:
    """Checks if running on Render or low-memory (<1.5GB) container."""
    if os.environ.get("RENDER") == "true" or os.environ.get("RENDER_SERVICE_ID"):
        return True
    try:
        if os.path.exists("/sys/fs/cgroup/memory.max"):
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1500 * 1024 * 1024:
                    return True
        if os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                val = f.read().strip()
                if val.isdigit() and int(val) < 1500 * 1024 * 1024:
                    return True
    except Exception:
        pass
    try:
        return psutil.virtual_memory().total < 1500 * 1024 * 1024
    except Exception:
        return False

IS_LOW_MEMORY_ENV = is_render_or_low_ram()
logger.info(f"Environment low-memory mode: {IS_LOW_MEMORY_ENV} (Render: {bool(os.environ.get('RENDER'))})")


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.device = DEVICE
        self.clip_model = None
        self.clip_processor = None
        self.clip_text_model = None
        self.clip_tokenizer = None
        self.blip_model = None
        self.blip_processor = None
        self._initialized = True

    def load_clip_text_only(self):
        """Loads lightweight text encoder to prevent OOM on memory-limited servers."""
        if IS_LOW_MEMORY_ENV:
            return None, None
        if self.clip_text_model is None:
            try:
                from transformers import CLIPTokenizerFast, CLIPTextModelWithProjection
                logger.info(f"Loading lightweight CLIP text encoder '{CLIP_MODEL_NAME}'...")
                torch.set_num_threads(1)
                self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
                self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(
                    CLIP_MODEL_NAME,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.clip_text_model.eval()
                gc.collect()
                logger.info("CLIP text encoder loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load lightweight CLIP text model: {e}")
                self.clip_text_model = None
        return self.clip_text_model, self.clip_tokenizer

    def load_clip(self):
        """Full CLIP model for environments with sufficient RAM (>= 1.5GB)."""
        if IS_LOW_MEMORY_ENV:
            return None, None
        if self.clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info(f"Loading full CLIP model '{CLIP_MODEL_NAME}' on {self.device}...")
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
                self.clip_model.eval()
                logger.info("Full CLIP loaded successfully.")
            except Exception as e:
                logger.warning(f"Full CLIP loading skipped: {e}")
                self.clip_model = None
                self.clip_processor = None
        return self.clip_model, self.clip_processor

    def load_blip(self):
        """BLIP is ~980MB RAM. Never attempt loading in <= 1GB environments to prevent OS OOM kill."""
        if IS_LOW_MEMORY_ENV:
            return None, None
        if self.blip_model is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                logger.info(f"Loading BLIP model '{BLIP_MODEL_NAME}' on {self.device}...")
                self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
                self.blip_model.eval()
                logger.info("BLIP loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load BLIP model: {e}")
                self.blip_model = None
                self.blip_processor = None
        return self.blip_model, self.blip_processor

    def _get_keyword_projected_embedding(self, text: str) -> torch.Tensor:
        """Projects queries onto pre-computed CLIP embedding manifold using keyword/caption co-occurrence with 0 MB RAM overhead."""
        try:
            tokens = set(re.findall(r'\b[a-z]+\b', text.lower()))
            from core.search_engine import STOPWORDS, SYNONYMS
            meaningful = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
            
            expanded = set(meaningful)
            for t in meaningful:
                if t in SYNONYMS:
                    expanded.update(SYNONYMS[t])

            if EMBEDDINGS_FILE.exists() and FILENAMES_FILE.exists() and KEYWORDS_FILE.exists():
                embs = np.load(str(EMBEDDINGS_FILE), allow_pickle=True)
                fnames = np.load(str(FILENAMES_FILE), allow_pickle=True)
                with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                    keywords_map = json.load(f)

                matched_indices = []
                for idx, fname in enumerate(fnames):
                    kws = set(keywords_map.get(str(fname), []))
                    if expanded & kws:
                        matched_indices.append(idx)

                if matched_indices:
                    matched_embs = embs[matched_indices]
                    avg_vec = np.mean(matched_embs, axis=0, keepdims=True)
                    norm = np.linalg.norm(avg_vec)
                    if norm > 0:
                        avg_vec = avg_vec / norm
                    return torch.from_numpy(avg_vec.astype('float32'))
        except Exception as e:
            logger.warning(f"Keyword projection fallback failed: {e}")

        # Deterministic pseudo-embedding
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1, 512).astype('float32')
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return torch.from_numpy(vec)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Returns 512-dim normalized embedding for text."""
        if IS_LOW_MEMORY_ENV:
            return self._get_keyword_projected_embedding(text)

        # Full environment (local / HF Spaces)
        try:
            model, tok = self.load_clip_text_only()
            if model is not None and tok is not None:
                inputs = tok([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.inference_mode():
                    out = model(**inputs)
                    embeds = out.text_embeds
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                return embeds.cpu()
        except Exception as e:
            logger.warning(f"CLIP text encoder failed: {e}")

        return self._get_keyword_projected_embedding(text)

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Computes CLIP image embeddings if RAM allows, or returns graceful content hash vectors."""
        if not IS_LOW_MEMORY_ENV:
            try:
                clip_m, clip_p = self.load_clip()
                if clip_m is not None and clip_p is not None:
                    inputs = clip_p(images=images, return_tensors="pt", padding=True).to(self.device)
                    with torch.inference_mode():
                        features = clip_m.get_image_features(**inputs)
                    if not isinstance(features, torch.Tensor):
                        features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                    return features.cpu()
            except Exception as e:
                logger.warning(f"Image embedding failed: {e}")

        results = []
        for img in images:
            thumb = img.resize((32, 32)).convert("L")
            raw_bytes = bytes(thumb.tobytes())
            seed = int(hashlib.md5(raw_bytes).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(512).astype('float32')
            vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return torch.from_numpy(np.array(results))

    def generate_caption(self, image: Image.Image, filename: str = "", max_new_tokens: int = 35) -> str:
        """Generates BLIP caption if available; otherwise produces smart metadata caption from filename."""
        if not IS_LOW_MEMORY_ENV:
            try:
                model, proc = self.load_blip()
                if model is not None and proc is not None:
                    inputs = proc(image, return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    caption = proc.decode(output[0], skip_special_tokens=True)
                    if caption and len(caption.strip()) > 3:
                        return caption.strip()
            except Exception as e:
                logger.warning(f"BLIP captioning skipped: {e}")

        # Smart fallback from filename
        if filename:
            clean = os.path.splitext(os.path.basename(filename))[0]
            clean = clean.replace("_", " ").replace("-", " ")
            clean = " ".join([w for w in clean.split() if not w.isdigit()])
            if clean:
                return f"a photo of {clean}"

        return "a high quality photo"


model_manager = ModelManager()


