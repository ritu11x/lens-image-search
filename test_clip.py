from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

print("CLIP model download ho raha hai... thoda time lagega pehli baar")

# Pretrained CLIP model load kar rahe hain
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Model load ho gaya!  ")

# Ek quick test - text embedding banate hain
text_inputs = processor(text=["a photo of a dog", "a photo of a cat"], return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# Agar tensor nahi hai, toh unwrap kar
if not isinstance(text_features, torch.Tensor):
    text_features = text_features.pooler_output if hasattr(text_features, 'pooler_output') else text_features[0]

print("Text embedding shape:", text_features.shape)