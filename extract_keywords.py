import json
import re

# Common English stopwords - inhe keywords mein nahi rakhna
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "on", "in", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "and",
    "or", "but", "if", "then", "there", "here", "this", "that", "these",
    "those", "it", "its", "as", "some", "there", "their", "his", "her"
}

with open("captions.json", "r") as f:
    captions = json.load(f)

keywords = {}

for filename, caption in captions.items():
    # Lowercase kar, punctuation hata, words mein split kar
    words = re.findall(r'\b[a-z]+\b', caption.lower())
    # Stopwords aur bahut chhote words (jaise "a", "of") hata
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    keywords[filename] = filtered

with open("keywords.json", "w") as f:
    json.dump(keywords, f, indent=2)

print(f"✅ Keywords extract ho gaye! Total images: {len(keywords)}")

# Sample check - pehli 3 images ke keywords dikha
for filename in list(keywords.keys())[:3]:
    print(f"\n{filename}")
    print(f"  Caption: {captions[filename]}")
    print(f"  Keywords: {keywords[filename]}")