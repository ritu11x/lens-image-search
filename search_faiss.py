from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import faiss
import time
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

embeddings = np.load("embeddings.npy").astype('float32')
filenames = np.load("filenames.npy")
embedding_dim = embeddings.shape[1]

faiss.normalize_L2(embeddings)

print("FAISS HNSW index bana rahe hain...")
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)
index.hnsw.efSearch = 64  # default 16 se badha diya
print(f"Index ready! Total vectors: {index.ntotal}")


def search_faiss(query_text, top_k=3):
    # CLIP encoding time (fixed cost, dono methods mein same)
    encode_start = time.time()
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs)

    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.pooler_output if hasattr(query_embedding, 'pooler_output') else query_embedding[0]

    query_embedding = query_embedding.squeeze().numpy().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    encode_time_ms = (time.time() - encode_start) * 1000

    # Pure FAISS search time
    search_start = time.time()
    distances, indices = index.search(query_embedding, top_k)
    pure_search_time_ms = (time.time() - search_start) * 1000

    print(f"\nQuery: '{query_text}'")
    print(f"  CLIP encoding: {encode_time_ms:.2f} ms")
    print(f"  Pure FAISS search: {pure_search_time_ms:.4f} ms")
    print("Top matches:")
    for idx in indices[0]:
        print(f"  {filenames[idx]}")


search_faiss("a photo of nature")
search_faiss("a person")
search_faiss("food")
search_faiss("nature landscape")
search_faiss("technology")