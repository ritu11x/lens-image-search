import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List, Union
import logging

from core.config import CLIP_MODEL_NAME, BLIP_MODEL_NAME, DEVICE

logger = logging.getLogger("lens.models")

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.device = DEVICE
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self._initialized = True

    def load_clip(self):
        if self.clip_model is None:
            logger.info(f"Loading CLIP model '{CLIP_MODEL_NAME}' on {self.device}...")
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
            self.clip_model.eval()
            logger.info("CLIP loaded successfully.")
        return self.clip_model, self.clip_processor

    def load_blip(self):
        if self.blip_model is None:
            logger.info(f"Loading BLIP model '{BLIP_MODEL_NAME}' on {self.device}...")
            self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(self.device)
            self.blip_model.eval()
            logger.info("BLIP loaded successfully.")
        return self.blip_model, self.blip_processor

    def get_text_embedding(self, text: str) -> torch.Tensor:
        self.load_clip()
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
        return features.cpu()

    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        self.load_clip()
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
        return features.cpu()

    def generate_caption(self, image: Image.Image, max_new_tokens: int = 35) -> str:
        self.load_blip()
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()


model_manager = ModelManager()
