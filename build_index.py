from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

image_folder = "images"
image_files = sorted(os.listdir(image_folder))  # sorted taaki order consistent rahe

all_embeddings = []

for filename in image_files:
    path = os.path.join(image_folder, filename)
    image = Image.open(path).convert("RGB")  # RGB mein convert - kuch PNGs transparent hote hain, isse error nahi aayega

    image_inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**image_inputs)

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]

    # numpy array mein convert kar aur list mein daal
    embedding = features.squeeze().numpy()
    all_embeddings.append(embedding)

    print(f"Processed: {filename}")

# Sabhi embeddings ko ek single numpy array mein combine kar
all_embeddings = np.array(all_embeddings)

# Save kar do - do files: embeddings aur unke corresponding filenames
np.save("embeddings.npy", all_embeddings)
np.save("filenames.npy", np.array(image_files))

print("\n✅ Sab embeddings save ho gaye!")
print("Shape:", all_embeddings.shape)  # expect: (5, 512) - 5 images, 512 numbers each