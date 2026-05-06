import torch
from PIL import Image
from transformers import TrOCRProcessor

from src.ocr.model import build_model


class OCRPredictor:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("📦 Loading processor...")
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        print("📦 Loading model...")
        self.model = build_model()

        print("📦 Loading weights...")
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        print("✅ Predictor ready")

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                num_beams=5,
                max_length=128
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text