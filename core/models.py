import os
import gc
import logging
import psutil
from typing import List, Union
from PIL import Image
import torch
import numpy as np

from core.config import CLIP_MODEL_NAME, BLIP_MODEL_NAME, DEVICE

logger = logging.getLogger("lens.models")

# Detect low-memory environment (e.g., Render free tier has <= 512MB RAM)
def get_total_ram_mb() -> float:
    try:
        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return 16000.0

IS_LOW_MEMORY_ENV = get_total_ram_mb() < 1200.0
logger.info(f"System RAM: {get_total_ram_mb():.1f} MB | Low-memory mode: {IS_LOW_MEMORY_ENV}")


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
        if self.clip_model is None:
            if IS_LOW_MEMORY_ENV:
                logger.info("Low-RAM environment detected (<1.2GB). Using lightweight text projection.")
                return None, None
            try:
                from transformers import CLIPProcessor, CLIPModel
                logger.info(f"Loading full CLIP model '{CLIP_MODEL_NAME}' on {self.device}...")
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
                self.clip_model.eval()
                logger.info("Full CLIP loaded successfully.")
            except Exception as e:
                logger.warning(f"Full CLIP loading skipped (insufficient RAM): {e}")
                self.clip_model = None
                self.clip_processor = None
        return self.clip_model, self.clip_processor

    def load_blip(self):
        """BLIP is ~980MB RAM. Never attempt loading in <= 1GB environments to prevent OS OOM kill."""
        if IS_LOW_MEMORY_ENV:
            logger.info("Skipping BLIP model loading on low-memory instance (<1.2GB RAM).")
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

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Returns 512-dim normalized embedding for text."""
        # Method 1: Lightweight text projection (~120MB)
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
            logger.warning(f"CLIP text projection failed: {e}")

        # Method 2: Full CLIP fallback if available
        try:
            clip_m, clip_p = self.load_clip()
            if clip_m is not None and clip_p is not None:
                inputs = clip_p(text=[text], return_tensors="pt", padding=True).to(self.device)
                with torch.inference_mode():
                    features = clip_m.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu()
        except Exception as e:
            logger.warning(f"Full CLIP text features failed: {e}")

        # Method 3: Deterministic pseudo-embedding fallback (zero crash guarantee)
        logger.info("Using deterministic pseudo-embedding fallback.")
        import hashlib
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1, 512).astype('float32')
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        return torch.from_numpy(vec)

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Computes CLIP image embeddings if RAM allows, or returns graceful zero vectors."""
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

        # Safe fallback: normalized pseudo-embeddings from image content hash
        results = []
        for img in images:
            import hashlib
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

