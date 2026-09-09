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

embeddings = np.load("embeddings.npy").astype('float32')
filenames = np.load("filenames.npy")
embedding_dim = embeddings.shape[1]

with open("keywords.json", "r") as f:
    keywords_data = json.load(f)

faiss.normalize_L2(embeddings)
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)
index.hnsw.efSearch = 64
print("Model load ho gaya! ✅\n")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "and",
    "or", "but", "if", "then", "there", "here", "this", "that", "these",
    "those", "it", "its", "as", "some", "their", "his", "her"
}


def search(query_text, top_k=10):
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs)
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]
    query_embedding = query_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [str(filenames[idx]) for idx in indices[0]]


def is_relevant(query, filename):
    """Automatic criterion: SAARE non-stopword query words image ke keywords mein hone chahiye"""
    query_words = set(re.findall(r'\b[a-z]+\b', query.lower())) - STOPWORDS
    image_keywords = set(k.lower() for k in keywords_data.get(filename, []))
    return query_words.issubset(image_keywords)


EVAL_QUERIES = [
    "a red car", "a person by the water", "food on a table", "a flower",
    "a dog", "a black and white photo", "a city street at night",
    "mountains and nature", "a person smiling", "technology or electronics",
]

eval_set = {}
p5_scores, p10_scores = [], []

for query in EVAL_QUERIES:
    results = search(query, top_k=10)
    relevant = [f for f in results if is_relevant(query, f)]
    eval_set[query] = {"candidates": results, "relevant": relevant}

    p5 = len([f for f in results[:5] if f in relevant]) / 5
    p10 = len([f for f in results[:10] if f in relevant]) / 10
    p5_scores.append(p5)
    p10_scores.append(p10)

    print(f"'{query}' → relevant: {len(relevant)}/10, P@5: {p5:.2f}, P@10: {p10:.2f}")

with open("eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)

avg_p5 = sum(p5_scores) / len(p5_scores)
avg_p10 = sum(p10_scores) / len(p10_scores)

print(f"\n{'='*50}")
print(f"Average Precision@5:  {avg_p5:.3f}")
print(f"Average Precision@10: {avg_p10:.3f}")
print(f"{'='*50}")