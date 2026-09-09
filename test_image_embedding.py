from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

print("Model load ho raha hai...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model load ho gaya! ✅")

# images folder ke andar sab images ke naam list kar
image_folder = "images"
image_files = os.listdir(image_folder)
print("Images mili:", image_files)

# Pehli image utha aur embedding banate hain
first_image_path = os.path.join(image_folder, image_files[0])
image = Image.open(first_image_path)

image_inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)

# Same fix jo text ke liye kiya tha
if not isinstance(image_features, torch.Tensor):
    image_features = image_features.pooler_output if hasattr(image_features, 'pooler_output') else image_features[0]

print("Image:", image_files[0])
print("Image embedding shape:", image_features.shape)
print("Sab kuch sahi kaam kar raha hai! 🎉")