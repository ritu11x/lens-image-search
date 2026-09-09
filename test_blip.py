from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

os.environ["HF_HUB_OFFLINE"] = "0"  # pehli baar download ke liye internet chahiye
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print("BLIP model download ho raha hai... thoda time lagega pehli baar")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model load ho gaya! ✅")

# Ek image test kar
image_path = os.path.join("images", os.listdir("images")[0])
image = Image.open(image_path).convert("RGB")

inputs = blip_processor(image, return_tensors="pt")
output = blip_model.generate(**inputs, max_new_tokens=30)
caption = blip_processor.decode(output[0], skip_special_tokens=True)

print(f"\nImage: {image_path}")
print(f"Caption: {caption}")