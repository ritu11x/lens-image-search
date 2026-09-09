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

with open("captions.json", "r") as f:
    captions_data = json.load(f)

faiss.normalize_L2(embeddings)
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)
index.hnsw.efSearch = 64
print("Model load ho gaya! ✅\n")


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

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "and",
    "or", "but", "if", "then", "there", "here", "this", "that", "these",
    "those", "it", "its", "as", "some", "their", "his", "her"
}


def suggest_relevant(query, results):
    query_words = set(re.findall(r'\b[a-z]+\b', query.lower())) - STOPWORDS
    suggested = []
    for i, filename in enumerate(results, 1):
        caption_words = set(re.findall(r'\b[a-z]+\b', captions_data.get(filename, "").lower())) - STOPWORDS
        if query_words & caption_words:
            suggested.append(i)
    return suggested

EVAL_QUERIES = [
    "a red car", "a person by the water", "food on a table", "a flower",
    "a dog", "a black and white photo", "a city street at night",
    "mountains and nature", "a person smiling", "technology or electronics",
]

eval_set = {}

print("=" * 70)
print("Har query ke top-10 results dikhenge, unke saath ek SUGGESTION hoga")
print("(jo captions mein query ke words se match karta hai).")
print("Bas Enter dabao suggestion accept karne ke liye,")
print("ya apne khud ke numbers type kar (comma-separated) override karne ke liye,")
print("ya 'none' agar genuinely koi relevant nahi hai.")
print("=" * 70)

for query in EVAL_QUERIES:
    print(f"\n\nQuery: '{query}'")
    print("-" * 50)
    results = search(query, top_k=10)
    suggested = suggest_relevant(query, results)

    for i, filename in enumerate(results, 1):
        caption = captions_data.get(filename, "no caption")
        marker = " ← suggested" if i in suggested else ""
        print(f"{i}. [{filename}] {caption}{marker}")

    suggested_str = ",".join(str(i) for i in suggested) if suggested else "none"
    user_input = input(f"\nRelevant [Enter = accept '{suggested_str}', or type numbers, or 'none']: ").strip().lower()

    if user_input == "":
        chosen = suggested
    elif user_input == "none":
        chosen = []
    else:
        try:
            chosen = [int(x.strip()) for x in user_input.split(",")]
        except ValueError:
            print("Galat input, suggestion accept kar rahe hain.")
            chosen = suggested

    relevant_filenames = [results[i - 1] for i in chosen if 1 <= i <= len(results)]
    eval_set[query] = {"candidates": results, "relevant": relevant_filenames}

with open("eval_set.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print(f"\n\n✅ Evaluation set save ho gaya! Total queries: {len(eval_set)}")