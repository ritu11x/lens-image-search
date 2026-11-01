import os
import re
import json
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from core.config import (
    IMAGES_DIR, THUMBNAILS_DIR,
    EMBEDDINGS_FILE, FILENAMES_FILE, CAPTIONS_FILE, KEYWORDS_FILE, INDEX_JSON_FILE
)
from core.models import model_manager
from core.search_engine import search_engine, STOPWORDS

logger = logging.getLogger("lens.indexer")

def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename)
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', base)
    return safe


class Indexer:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {
            "status": "idle",
            "processed": 0,
            "total": 0,
            "message": "",
            "start_time": None
        }

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def extract_keywords_from_caption(self, caption: str) -> List[str]:
        words = re.findall(r'\b[a-z]+\b', caption.lower())
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
        # preserve order while removing duplicates
        seen = set()
        result = []
        for w in filtered:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result

    def generate_thumbnail(self, image: Image.Image, save_path: Path, max_size: int = 400):
        try:
            thumb = image.copy()
            thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            thumb.save(save_path, quality=85, optimize=False)
        except Exception as e:
            logger.warning(f"Could not generate thumbnail for {save_path}: {e}")

    def save_atomic(self, embeddings: np.ndarray, filenames: np.ndarray, captions: dict, keywords: dict):
        temp_emb = str(EMBEDDINGS_FILE) + ".tmp.npy"
        temp_names = str(FILENAMES_FILE) + ".tmp.npy"
        temp_caps = str(CAPTIONS_FILE) + ".tmp.json"
        temp_kws = str(KEYWORDS_FILE) + ".tmp.json"
        temp_idx = str(INDEX_JSON_FILE) + ".tmp.json"

        np.save(temp_emb, embeddings)
        np.save(temp_names, filenames)
        with open(temp_caps, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        with open(temp_kws, "w", encoding="utf-8") as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False)

        # Build portable index_with_keywords.json (Project Report Module 2 & 5)
        index_entries = []
        for i, fname in enumerate(filenames):
            fn_str = str(fname)
            emb_list = embeddings[i].tolist() if i < len(embeddings) else []
            index_entries.append({
                "filename": fn_str,
                "caption": captions.get(fn_str, ""),
                "keywords": keywords.get(fn_str, []),
                "embedding": emb_list
            })
        with open(temp_idx, "w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2, ensure_ascii=False)

        # Atomic replace
        os.replace(temp_emb, str(EMBEDDINGS_FILE))
        os.replace(temp_names, str(FILENAMES_FILE))
        os.replace(temp_caps, str(CAPTIONS_FILE))
        os.replace(temp_kws, str(KEYWORDS_FILE))
        os.replace(temp_idx, str(INDEX_JSON_FILE))

    def index_files(self, new_filenames: List[str], batch_size: int = 8) -> Dict[str, Any]:
        """Indexes specific files or newly discovered files."""
        with self.lock:
            self.status = {
                "status": "running",
                "processed": 0,
                "total": len(new_filenames),
                "message": f"Indexing {len(new_filenames)} images...",
                "start_time": time.time()
            }

        valid_files = []
        new_embeddings_list = []
        skipped = []

        try:
            # Existing data
            existing_emb = search_engine.embeddings if len(search_engine.embeddings) > 0 else np.empty((0, 512), dtype='float32')
            existing_names = search_engine.filenames.tolist() if len(search_engine.filenames) > 0 else []
            existing_captions = dict(search_engine.captions)
            existing_keywords = dict(search_engine.keywords)

            # Process in batches for CLIP embedding
            for i in range(0, len(new_filenames), batch_size):
                batch_files = new_filenames[i : i + batch_size]
                batch_images = []
                batch_valid_names = []

                for fname in batch_files:
                    path = IMAGES_DIR / fname
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images.append(img)
                        batch_valid_names.append(fname)

                        # Generate thumbnail
                        thumb_path = THUMBNAILS_DIR / fname
                        self.generate_thumbnail(img, thumb_path)

                        # BLIP Caption & Keywords (with smart filename fallback on low-memory)
                        caption = model_manager.generate_caption(img, filename=fname)
                        kws = self.extract_keywords_from_caption(caption)
                        existing_captions[fname] = caption
                        existing_keywords[fname] = kws

                    except Exception as err:
                        logger.error(f"Failed processing image {fname}: {err}")
                        skipped.append(fname)

                if batch_images:
                    feats = model_manager.get_image_embeddings(batch_images)
                    embs = feats.squeeze().numpy().astype('float32')
                    if embs.ndim == 1:
                        embs = embs.reshape(1, -1)
                    for emb, fname in zip(embs, batch_valid_names):
                        new_embeddings_list.append(emb)
                        valid_files.append(fname)

                with self.lock:
                    self.status["processed"] = min(len(new_filenames), i + len(batch_files))

            # Combine and save without duplicate filenames
            if valid_files:
                existing_names_list = [str(x) for x in existing_names]
                existing_name_to_idx = {name: idx for idx, name in enumerate(existing_names_list)}
                
                new_embs_to_append = []
                new_names_to_append = []
                
                # Make a mutable copy of existing embeddings
                if len(existing_emb) > 0:
                    combined_embs_list = list(existing_emb)
                    combined_names_list = list(existing_names_list)
                else:
                    combined_embs_list = []
                    combined_names_list = []
                
                for emb, fname in zip(new_embeddings_list, valid_files):
                    if fname in existing_name_to_idx:
                        idx = existing_name_to_idx[fname]
                        combined_embs_list[idx] = emb
                    else:
                        combined_embs_list.append(emb)
                        combined_names_list.append(fname)
                        existing_name_to_idx[fname] = len(combined_names_list) - 1

                combined_embs = np.array(combined_embs_list, dtype='float32') if combined_embs_list else np.empty((0, 512), dtype='float32')
                combined_names = np.array(combined_names_list, dtype=object)

                self.save_atomic(combined_embs, combined_names, existing_captions, existing_keywords)
                # Reload search engine with new data
                search_engine.load_index()

            with self.lock:
                skip_msg = f" (Skipped: {len(skipped)})" if skipped else ""
                self.status = {
                    "status": "done",
                    "processed": len(valid_files),
                    "total": len(new_filenames),
                    "message": f"Successfully indexed {len(valid_files)} image(s){skip_msg}.",
                    "start_time": None
                }

            return {
                "success": True,
                "indexed_count": len(valid_files),
                "skipped": skipped,
                "total_index_size": search_engine.index.ntotal if search_engine.index else 0
            }

        except Exception as e:
            logger.exception("Error during index build")
            with self.lock:
                self.status = {
                    "status": "error",
                    "processed": len(valid_files),
                    "total": len(new_filenames),
                    "message": f"Indexing error: {str(e)}",
                    "start_time": None
                }
            return {"success": False, "error": str(e)}

    def build_index_for_new_images(self) -> Dict[str, Any]:
        """Finds any images in images/ that are not yet in the index and indexes them."""
        all_disk = sorted([
            f.name for f in IMAGES_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ])
        indexed = set(search_engine.filenames.tolist() if len(search_engine.filenames) > 0 else [])
        unindexed = [f for f in all_disk if f not in indexed]

        if not unindexed:
            with self.lock:
                self.status = {
                    "status": "done",
                    "processed": 0,
                    "total": 0,
                    "message": "All images in images/ are already indexed.",
                    "start_time": None
                }
            return {"success": True, "message": "All images already indexed.", "indexed_count": 0}

        return self.index_files(unindexed)

    def reset_and_rebuild_base_index(self) -> Dict[str, Any]:
        """Privacy feature: scrubs all user uploads, leaves only BASE_SAMPLE_IMAGES, and rebuilds index."""
        from core.config import BASE_SAMPLE_IMAGES
        with self.lock:
            self.status = {
                "status": "running",
                "processed": 0,
                "total": 0,
                "message": "Scrubbing user uploaded photos and wiping privacy cache...",
                "start_time": time.time()
            }

        deleted_files = []
        for f in IMAGES_DIR.glob("*"):
            if f.is_file() and f.name not in BASE_SAMPLE_IMAGES:
                try:
                    f.unlink(missing_ok=True)
                    thumb = THUMBNAILS_DIR / f.name
                    thumb.unlink(missing_ok=True)
                    deleted_files.append(f.name)
                except Exception as err:
                    logger.warning(f"Could not delete {f.name}: {err}")

        # Filter in-memory and disk structures to only BASE_SAMPLE_IMAGES
        try:
            cur_names = search_engine.filenames.tolist() if len(search_engine.filenames) > 0 else []
            keep_indices = [i for i, name in enumerate(cur_names) if name in BASE_SAMPLE_IMAGES]

            if keep_indices:
                filtered_emb = search_engine.embeddings[keep_indices]
                filtered_names = np.array([cur_names[i] for i in keep_indices])
            else:
                filtered_emb = np.empty((0, 512), dtype='float32')
                filtered_names = np.array([])

            filtered_caps = {k: v for k, v in search_engine.captions.items() if k in BASE_SAMPLE_IMAGES}
            filtered_kws = {k: v for k, v in search_engine.keywords.items() if k in BASE_SAMPLE_IMAGES}

            self.save_atomic(filtered_emb, filtered_names, filtered_caps, filtered_kws)
            search_engine.load_index()

            with self.lock:
                self.status = {
                    "status": "done",
                    "processed": len(deleted_files),
                    "total": len(deleted_files),
                    "message": f"Privacy wipe complete: Permanently removed {len(deleted_files)} file(s).",
                    "start_time": None
                }

            return {
                "success": True,
                "deleted_count": len(deleted_files),
                "deleted_files": deleted_files,
                "total_remaining": search_engine.index.ntotal if search_engine.index else 0
            }
        except Exception as e:
            logger.exception("Error during privacy scrub")
            return {"success": False, "error": str(e)}


indexer = Indexer()

