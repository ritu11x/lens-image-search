import re
import math
import json
import threading
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import numpy as np
import faiss
from PIL import Image

from core.config import (
    EMBEDDINGS_FILE, FILENAMES_FILE, CAPTIONS_FILE, KEYWORDS_FILE,
    EMBEDDING_DIM, HNSW_M, HNSW_EF_SEARCH,
    DEFAULT_TOP_K, DEFAULT_CLIP_WEIGHT, DEFAULT_KEYWORD_WEIGHT, DEFAULT_CANDIDATE_POOL
)
from core.models import model_manager

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "and",
    "or", "but", "if", "then", "there", "here", "this", "that", "these",
    "those", "it", "its", "as", "some", "their", "his", "her", "photo", "image", "picture"
}

SYNONYMS = {
    "chair": ["chairs", "seat", "seats", "seating", "stool", "stools", "bench", "furniture", "table", "tables"],
    "chairs": ["chair", "seats", "seating", "stools", "furniture", "tables", "dining"],
    "table": ["tables", "desk", "counter", "chair", "chairs", "furniture", "dining", "cafe"],
    "tables": ["table", "desks", "chairs", "furniture", "dining", "cafe"],
    "restaurant": ["cafe", "diner", "bistro", "dining", "eatery", "canteen", "cafeteria", "room", "table", "chair"],
    "cafe": ["restaurant", "coffee", "bistro", "diner", "cafeteria", "tables", "chairs"],
    "room": ["indoor", "interior", "hall", "space", "dining", "restaurant"],
    "wall": ["walls", "tiles", "tiled", "backdrop"],
    "furniture": ["chair", "chairs", "table", "tables", "seat", "desk"],
    "person": ["man", "woman", "boy", "girl", "people", "guy", "lady", "human", "child"],
    "people": ["person", "men", "women", "crowd", "children"],
    "car": ["automobile", "vehicle", "sedan", "suv", "sports", "coupe"],
    "dog": ["puppy", "canine", "pet", "hound"],
    "cat": ["kitten", "feline", "pet"],
    "food": ["meal", "dish", "breakfast", "lunch", "dinner", "dessert", "fruit", "bread", "coffee", "pastry"],
    "nature": ["landscape", "mountain", "forest", "lake", "river", "trees", "outdoors", "sky"],
    "technology": ["electronics", "computer", "laptop", "screen", "keyboard", "phone", "gadget", "device"],
    "ocean": ["water", "sea", "beach", "coast", "waves"],
    "smile": ["smiling", "happy", "laughing", "cheerful"],
}


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avg_doc_len: float = 1.0
        self.doc_freq: Counter = Counter()
        self.corpus_size: int = 0

    def fit(self, keywords_data: Dict[str, List[str]]):
        self.corpus_size = len(keywords_data)
        if self.corpus_size == 0:
            self.avg_doc_len = 1.0
            return

        total_len = 0
        self.doc_freq = Counter()
        self.doc_len = {}

        for doc_id, kws in keywords_data.items():
            self.doc_len[doc_id] = len(kws)
            total_len += len(kws)
            unique_terms = set(kws)
            for term in unique_terms:
                self.doc_freq[term] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 1.0

    def score(self, query_tokens: Set[str], doc_tokens: List[str], doc_id: str) -> float:
        if not query_tokens or not doc_tokens or self.corpus_size == 0:
            return 0.0

        score = 0.0
        doc_len = self.doc_len.get(doc_id, len(doc_tokens))
        term_counts = Counter(doc_tokens)

        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue

            df = self.doc_freq.get(token, 0)
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
            score += idf * (numerator / denominator)

        # Normalize to roughly [0, 1] range based on max possible score for query
        max_possible = sum(math.log(1.0 + (self.corpus_size + 0.5) / 0.5) for _ in query_tokens) or 1.0
        return min(1.0, score / max_possible) if max_possible > 0 else 0.0


class SearchEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.bm25 = BM25Scorer()
        self.embeddings: np.ndarray = np.empty((0, EMBEDDING_DIM), dtype='float32')
        self.filenames: np.ndarray = np.empty((0,), dtype=str)
        self.captions: Dict[str, str] = {}
        self.keywords: Dict[str, List[str]] = {}
        self.index: Optional[faiss.IndexHNSWFlat] = None
        self.load_index()

    def load_index(self):
        with self.lock:
            if EMBEDDINGS_FILE.exists() and FILENAMES_FILE.exists():
                try:
                    self.embeddings = np.load(str(EMBEDDINGS_FILE)).astype('float32')
                    self.filenames = np.load(str(FILENAMES_FILE), allow_pickle=True)
                except Exception as e:
                    logger.error(f"Error loading embeddings or filenames: {e}")
                    self.embeddings = np.empty((0, EMBEDDING_DIM), dtype='float32')
                    self.filenames = np.empty((0,), dtype=str)
            else:
                self.embeddings = np.empty((0, EMBEDDING_DIM), dtype='float32')
                self.filenames = np.empty((0,), dtype=str)

            if CAPTIONS_FILE.exists():
                try:
                    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
                        self.captions = json.load(f)
                except Exception:
                    self.captions = {}
            else:
                self.captions = {}

            if KEYWORDS_FILE.exists():
                try:
                    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                        self.keywords = json.load(f)
                except Exception:
                    self.keywords = {}
            else:
                self.keywords = {}

            # Build FAISS HNSW Index
            self.index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M)
            self.index.hnsw.efSearch = HNSW_EF_SEARCH

            if len(self.embeddings) > 0:
                normalized = self.embeddings.copy()
                faiss.normalize_L2(normalized)
                self.index.add(normalized)

            self.bm25.fit(self.keywords)

    def extract_query_keywords(self, query: str, expand_synonyms: bool = True) -> Set[str]:
        words = re.findall(r'\b[a-z]+\b', query.lower())
        base_tokens = set()
        for w in words:
            if w in STOPWORDS or len(w) <= 2:
                continue
            base_tokens.add(w)
            # Automatic singular / plural stemming
            if w.endswith('ies') and len(w) > 4:
                base_tokens.add(w[:-3] + 'y')
            elif w.endswith('es') and len(w) > 4:
                base_tokens.add(w[:-2])
            elif w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
                base_tokens.add(w[:-1])

        if not expand_synonyms:
            return base_tokens

        expanded = set(base_tokens)
        for token in base_tokens:
            if token in SYNONYMS:
                expanded.update(SYNONYMS[token])
        return expanded

    def search_text(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        clip_weight: float = DEFAULT_CLIP_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        min_score: float = 0.0,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
        negative_query: Optional[str] = None,
        negative_weight: float = 0.45
    ) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            pool_size = min(candidate_pool, self.index.ntotal)
            if pool_size <= 0:
                return []

            # Check for inline negative syntax (e.g. "cars --without red" or "car -not red")
            pos_query = query.strip()
            neg_query = (negative_query or "").strip()

            inline_match = re.split(r'\s+(?:--without|--not|--no|-without|-not)\s+', pos_query, maxsplit=1, flags=re.IGNORECASE)
            if len(inline_match) == 2:
                pos_query = inline_match[0].strip()
                if not neg_query:
                    neg_query = inline_match[1].strip()

            templated_pos = pos_query if pos_query.lower().startswith("a photo of") else f"a photo of {pos_query}"
            pos_feat = model_manager.get_text_embedding(templated_pos)
            pos_emb = pos_feat.squeeze().numpy().astype('float32').reshape(1, -1)
            faiss.normalize_L2(pos_emb)

            # Feature 2: Negative Vector Subtraction
            if neg_query:
                templated_neg = neg_query if neg_query.lower().startswith("a photo of") else f"a photo of {neg_query}"
                neg_feat = model_manager.get_text_embedding(templated_neg)
                neg_emb = neg_feat.squeeze().numpy().astype('float32').reshape(1, -1)
                faiss.normalize_L2(neg_emb)

                # Linear algebra vector steering: v_query = v_pos - lambda * v_neg
                steered_emb = pos_emb - (negative_weight * neg_emb)
                faiss.normalize_L2(steered_emb)
                query_emb = steered_emb
            else:
                query_emb = pos_emb

            distances, indices = self.index.search(query_emb, pool_size)
            query_kws = self.extract_query_keywords(pos_query, expand_synonyms=True)
            neg_kws = self.extract_query_keywords(neg_query, expand_synonyms=True) if neg_query else set()

            scored = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.filenames):
                    continue

                fname = str(self.filenames[idx])
                dist = distances[0][i]
                clip_sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))

                doc_kws = self.keywords.get(fname, [])
                bm25_score = self.bm25.score(query_kws, doc_kws, fname)

                # Keyword overlap
                direct_tokens = self.extract_query_keywords(pos_query, expand_synonyms=False)
                overlap = list(direct_tokens & set(doc_kws))

                # Negative keyword penalty: heavily penalize photos matching excluded concepts
                if neg_kws:
                    neg_overlap = neg_kws & set(doc_kws)
                    if neg_overlap:
                        bm25_score = max(0.0, bm25_score - 0.5)
                        clip_sim = max(0.0, clip_sim - 0.15)

                final_score = (clip_weight * clip_sim) + (keyword_weight * bm25_score)

                # Strict relevance check
                is_keyword_matched = (bm25_score > 0.05) or (len(overlap) > 0)
                is_clip_match = clip_sim >= max(0.20, min_score)

                # Keyword-matched photos receive additive boost over visual similarity
                if is_keyword_matched:
                    final_score = max(final_score, clip_sim + (0.20 * bm25_score) + (0.05 * len(overlap)))
                elif is_clip_match:
                    final_score = max(final_score, clip_sim)

                if bool(query_kws) and not is_keyword_matched and not is_clip_match:
                    continue

                if (final_score >= min_score) or (clip_sim >= max(0.20, min_score)):
                    scored.append({
                        "filename": fname,
                        "caption": self.captions.get(fname, ""),
                        "keywords": doc_kws[:8],
                        "matched_keywords": overlap,
                        "clip_score": round(clip_sim, 4),
                        "keyword_score": round(bm25_score, 4),
                        "final_score": round(final_score, 4),
                        "is_keyword_matched": is_keyword_matched
                    })

            scored.sort(key=lambda x: x["final_score"], reverse=True)
            return scored[:top_k]

    def search_composed(
        self,
        image: Optional[Image.Image] = None,
        base_filename: Optional[str] = None,
        text_modifier: str = "",
        image_weight: float = 0.5,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Feature 1: Composed Multimodal Retrieval (Image + Text Refinement)."""
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            # Step 1: Base image embedding
            if image is not None:
                img_rgb = image.convert("RGB")
                img_feat = model_manager.get_image_embeddings([img_rgb])
                img_emb = img_feat.squeeze().numpy().astype('float32').reshape(1, -1)
            elif base_filename is not None:
                idx_list = np.where(self.filenames == base_filename)[0]
                if len(idx_list) == 0:
                    return []
                img_emb = self.embeddings[idx_list[0]:idx_list[0]+1].copy()
            else:
                return []

            faiss.normalize_L2(img_emb)

            # Step 2: Text modifier embedding
            clean_mod = text_modifier.strip()
            if clean_mod:
                templated = clean_mod if clean_mod.lower().startswith("a photo of") else f"a photo of {clean_mod}"
                text_feat = model_manager.get_text_embedding(templated)
                text_emb = text_feat.squeeze().numpy().astype('float32').reshape(1, -1)
                faiss.normalize_L2(text_emb)

                # Step 3: Multimodal Composition Arithmetic
                alpha = max(0.1, min(0.9, float(image_weight)))
                composed_emb = (alpha * img_emb) + ((1.0 - alpha) * text_emb)
                faiss.normalize_L2(composed_emb)
            else:
                composed_emb = img_emb

            pool_size = min(top_k + (1 if base_filename else 0), self.index.ntotal)
            distances, indices = self.index.search(composed_emb, pool_size)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.filenames):
                    continue
                fname = str(self.filenames[idx])
                if base_filename and fname == base_filename:
                    continue  # skip self

                dist = distances[0][i]
                sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                results.append({
                    "filename": fname,
                    "caption": self.captions.get(fname, ""),
                    "keywords": self.keywords.get(fname, [])[:8],
                    "similarity": round(sim, 4),
                    "is_composed": bool(clean_mod),
                    "text_modifier": clean_mod
                })
                if len(results) >= top_k:
                    break

            return results

    def search_image(self, image: Image.Image, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            pool_size = min(top_k, self.index.ntotal)
            image_rgb = image.convert("RGB")
            feat = model_manager.get_image_embeddings([image_rgb])
            query_emb = feat.squeeze().numpy().astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_emb)

            distances, indices = self.index.search(query_emb, pool_size)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.filenames):
                    continue
                fname = str(self.filenames[idx])
                dist = distances[0][i]
                sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                results.append({
                    "filename": fname,
                    "caption": self.captions.get(fname, ""),
                    "keywords": self.keywords.get(fname, [])[:8],
                    "similarity": round(sim, 4)
                })
            return results

    def find_similar(self, filename: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        with self.lock:
            if self.index is None or self.index.ntotal <= 1:
                return []

            try:
                idx_list = np.where(self.filenames == filename)[0]
                if len(idx_list) == 0:
                    return []
                target_idx = idx_list[0]
                target_emb = self.embeddings[target_idx:target_idx+1].copy()
                faiss.normalize_L2(target_emb)

                # Search top_k + 1 to exclude self
                pool_size = min(top_k + 1, self.index.ntotal)
                distances, indices = self.index.search(target_emb, pool_size)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.filenames):
                        continue
                    fname = str(self.filenames[idx])
                    if fname == filename:
                        continue  # skip self
                    dist = distances[0][i]
                    sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                    results.append({
                        "filename": fname,
                        "caption": self.captions.get(fname, ""),
                        "keywords": self.keywords.get(fname, [])[:8],
                        "similarity": round(sim, 4)
                    })
                    if len(results) >= top_k:
                        break
                return results
            except Exception as e:
                return []


search_engine = SearchEngine()
