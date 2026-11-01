import os
import gc
import json
import re
import hashlib
import logging
from typing import List, Union, Optional
from PIL import Image
import torch
import numpy as np

from core.config import (
    CLIP_MODEL_NAME, BLIP_MODEL_NAME, DEVICE,
    EMBEDDINGS_FILE, FILENAMES_FILE, KEYWORDS_FILE, CAPTIONS_FILE
)

logger = logging.getLogger("lens.models")

def is_low_memory_env() -> bool:
    """Safe-by-default for cloud hosts (Render 512MB). On local machines or HF Spaces (16GB), runs full CLIP and BLIP models."""
    if os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"):
        return True
    if os.environ.get("LOW_MEMORY_MODE", "").lower() == "true":
        return True
    try:
        import psutil
        if psutil.virtual_memory().total < 1800 * 1024 * 1024:
            return True
    except Exception:
        pass
    return False

IS_LOW_MEMORY_ENV = is_low_memory_env()
logger.info(f"Low-memory manifold mode active: {IS_LOW_MEMORY_ENV}")


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
        self.vocab_embeddings = None
        self.vocab_words = []
        self._load_vocab()
        self._initialized = True

    def _load_vocab(self):
        from core.config import VOCAB_EMBEDDINGS_FILE, VOCAB_WORDS_FILE
        if VOCAB_EMBEDDINGS_FILE.exists() and VOCAB_WORDS_FILE.exists():
            try:
                self.vocab_embeddings = np.load(str(VOCAB_EMBEDDINGS_FILE)).astype('float32')
                norms = np.linalg.norm(self.vocab_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.vocab_embeddings = self.vocab_embeddings / norms
                with open(VOCAB_WORDS_FILE, "r", encoding="utf-8") as f:
                    self.vocab_words = json.load(f)
                logger.info(f"Loaded visual concept vocabulary ({len(self.vocab_words)} concepts).")
            except Exception as e:
                logger.warning(f"Could not load visual vocab: {e}")

    def detect_visual_concepts(self, image_embeddings: Union[np.ndarray, torch.Tensor], top_k: int = 7, min_sim: float = 0.20) -> List[List[str]]:
        """Instantaneously (1ms) detects multi-label visual concepts for batch of image embeddings via matrix projection."""
        if self.vocab_embeddings is None or not self.vocab_words:
            self._load_vocab()
        if self.vocab_embeddings is None or not self.vocab_words:
            count = len(image_embeddings) if hasattr(image_embeddings, '__len__') else 1
            return [[] for _ in range(count)]
        
        if isinstance(image_embeddings, torch.Tensor):
            embs = image_embeddings.detach().cpu().numpy().astype('float32')
        else:
            embs = np.array(image_embeddings, dtype='float32')
        
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if len(embs) == 0:
            return []

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms

        sims = embs @ self.vocab_embeddings.T  # [B, N_vocab]
        results = []
        for row in sims:
            top_indices = np.argsort(row)[::-1]
            detected = [self.vocab_words[idx] for idx in top_indices if row[idx] >= min_sim][:top_k]
            results.append(detected)
        return results

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
                text_clean = text.strip()
                words = text_clean.split()
                if len(words) <= 3 and not text_clean.lower().startswith(("a ", "an ", "the ", "photo of")):
                    query_prompt = f"a photo of {text_clean}"
                else:
                    query_prompt = text_clean

                inputs = tok([query_prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
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

    def generate_caption(
        self,
        image: Image.Image,
        filename: str = "",
        visual_tags: Optional[List[str]] = None,
        skip_blip: bool = False,
        max_new_tokens: int = 35
    ) -> str:
        """Generates rich, descriptive caption using BLIP + automatic multi-label concept enrichment."""
        blip_caption = ""
        if not IS_LOW_MEMORY_ENV and not skip_blip:
            try:
                model, proc = self.load_blip()
                if model is not None and proc is not None:
                    inputs = proc(image, return_tensors="pt").to(self.device)
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=min(max_new_tokens, 20), num_beams=1)
                    caption = proc.decode(output[0], skip_special_tokens=True)
                    if caption and len(caption.strip()) > 3:
                        blip_caption = caption.strip()
            except Exception as e:
                logger.warning(f"BLIP captioning skipped: {e}")

        tags = [t for t in (visual_tags or []) if t]

        # If BLIP caption is present, enrich it with any detected primary visual concepts it missed
        if blip_caption:
            clean_blip = blip_caption.lower()
            missing = [t for t in tags if t not in clean_blip][:4]
            if missing:
                return f"{blip_caption} with {', '.join(missing)}"
            return blip_caption

        # Ultra-fast synthesis from multi-label visual concepts (1ms for bulk 100-photo uploads)
        if tags:
            tag_str = ", ".join(tags[:5])
            return f"a photo of {tag_str}"

        # Fallback from filename
        if filename:
            clean = os.path.splitext(os.path.basename(filename))[0]
            clean = clean.replace("_", " ").replace("-", " ")
            clean = " ".join([w for w in clean.split() if not w.isdigit()])
            if clean:
                return f"a photo of {clean}"

        return "a high quality photo"


model_manager = ModelManager()


