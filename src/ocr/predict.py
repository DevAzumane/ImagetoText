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

    def predict(self, image_path, return_confidence=False):
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,

                # 🔥 CRITICAL FIX (match training)
                num_beams=1,                  # remove language hallucination
                do_sample=False,
                early_stopping=True,
                no_repeat_ngram_size=2,
                max_length=64,

                # 🔥 confidence
                output_scores=True,
                return_dict_in_generate=True
            )

        generated_ids = outputs.sequences

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # 🔥 simple confidence score
        confidence = None
        if return_confidence:
            scores = outputs.scores
            if scores:
                avg_score = torch.mean(torch.stack([
                    torch.max(s, dim=-1).values.mean() for s in scores
                ]))
                confidence = float(avg_score)

        return text, confidence if return_confidence else text