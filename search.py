from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import time
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

embeddings = np.load("embeddings.npy")
filenames = np.load("filenames.npy")


def search(query_text, top_k=3):
    # CLIP encoding time (fixed cost)
    encode_start = time.time()
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs)

    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]

    query_embedding = query_embedding.squeeze().numpy()
    encode_time_ms = (time.time() - encode_start) * 1000

    # Pure brute-force search time
    search_start = time.time()
    similarities = []
    for img_emb in embeddings:
        dot_product = np.dot(query_embedding, img_emb)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(img_emb)
        similarity = dot_product / norm_product
        similarities.append(similarity)

    similarities = np.array(similarities)
    top_indices = similarities.argsort()[::-1][:top_k]
    pure_search_time_ms = (time.time() - search_start) * 1000

    print(f"\nQuery: '{query_text}'")
    print(f"  CLIP encoding: {encode_time_ms:.2f} ms")
    print(f"  Pure brute-force search: {pure_search_time_ms:.4f} ms")
    print("Top matches:")
    for idx in top_indices:
        print(f"  {filenames[idx]}  (similarity: {similarities[idx]:.4f})")


search("a photo of nature")
search("a person")
search("food")
search("nature landscape")
search("technology")