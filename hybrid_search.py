from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import faiss
import json
import re
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

# Saara data load kar
embeddings = np.load("embeddings.npy").astype('float32')
filenames = np.load("filenames.npy")
embedding_dim = embeddings.shape[1]

with open("keywords.json", "r") as f:
    keywords_data = json.load(f)

faiss.normalize_L2(embeddings)
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)
index.hnsw.efSearch = 64

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
    return len(overlap) / len(query_keywords)  # kitne query keywords match hue, 0 to 1


def hybrid_search(query_text, top_k=6, clip_weight=0.8, keyword_weight=0.2, candidate_pool=50):
    # Step 1: CLIP se query embedding banao
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs)
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]
    query_embedding = query_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    # Step 2: FAISS se ek bada candidate pool nikaal (top_k se zyada, taaki rerank karne ke liye options hon)
    distances, indices = index.search(query_embedding, candidate_pool)

    # Step 3: Query se keywords nikaal
    query_keywords = extract_query_keywords(query_text)

    # Step 4: Har candidate ke liye hybrid score calculate kar
    scored_results = []
    for i, idx in enumerate(indices[0]):
        filename = str(filenames[idx])
        clip_distance = distances[0][i]
        clip_similarity = 1 - (clip_distance / 2)  # normalized L2 -> similarity

        image_keywords = keywords_data.get(filename, [])
        kw_score = keyword_match_score(query_keywords, image_keywords)

        final_score = (clip_weight * clip_similarity) + (keyword_weight * kw_score)

        scored_results.append({
    "filename": filename,
    "clip_score": round(float(clip_similarity), 4),
    "keyword_score": round(kw_score, 4),
    "final_score": round(float(final_score), 4)   # yahan float() add kar diya
})

    # Step 5: Final score ke hisaab se re-sort kar (candidate pool ke andar)
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    return scored_results[:top_k]


# Test kar
if __name__ == "__main__":
    results = hybrid_search("a red car", top_k=5, clip_weight=0.8, keyword_weight=0.2)
    for r in results:
        print(r)