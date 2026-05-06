import os

from src.ocr.predict import OCRPredictor

def main():
    print("🚀 STARTING INFERENCE")

    model_path = "models/ocr/model.pth"
    image_dir = "data/lines"

    predictor = OCRPredictor(model_path)

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    print(f"\n📄 Found {len(images)} images\n")

    for img_name in images[:30]:  # limit for quick test
        img_path = os.path.join(image_dir, img_name)

        result = predictor.predict(img_path)

        print(f"📷 {img_name}")
        print(f"🧠 {result}\n")

if __name__ == "__main__":
    main()