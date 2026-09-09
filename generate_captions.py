from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("BLIP model load ho raha hai...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model load ho gaya! ✅")

image_folder = "images"
image_files = sorted(os.listdir(image_folder))

captions = {}

for i, filename in enumerate(image_files):
    path = os.path.join(image_folder, filename)
    image = Image.open(path).convert("RGB")

    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    captions[filename] = caption

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(image_files)} images...")

# JSON file mein save kar - filename -> caption mapping
with open("captions.json", "w") as f:
    json.dump(captions, f, indent=2)

print(f"\n✅ Sab captions save ho gaye! Total: {len(captions)}")