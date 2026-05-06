import os

from src.ocr.predict import OCRPredictor


def main():
    print("🚀 STARTING INFERENCE")

    model_path = "models/ocr/model.pth"
    image_dir = "data/lines"

    predictor = OCRPredictor(model_path)

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    print(f"\n📄 Found {len(images)} images\n")

    for img_name in images[:5]:  # test few samples
        img_path = os.path.join(image_dir, img_name)

        text, conf = predictor.predict(img_path, return_confidence=True)

        print(f"📷 {img_name}")
        print(f"🧠 {text}")
        print(f"📊 Confidence: {conf:.4f}" if conf else "")
        print("-" * 50)


if __name__ == "__main__":
    main()