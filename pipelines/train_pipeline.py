import os
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
from PIL import Image

from src.ocr.model import build_model
from src.ocr.dataset import LineDataset
from src.ocr.train import train_one_epoch, predict_sample


def debug_predictions(model, processor, dataset, device, num_samples=3):
    print("\n🧠 SAMPLE PREDICTIONS:")

    for i in range(min(num_samples, len(dataset))):
        img_name, gt = dataset.samples[i]

        img_path = os.path.join(dataset.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        pred = predict_sample(model, processor, img, device)

        print(f"\n📷 {img_name}")
        print(f"GT   : {gt}")
        print(f"PRED : {pred}")


def main():
    print("🚀 STARTING TRAINING")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    image_dir = "data/lines"
    label_file = "data/labels/labels.txt"

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    dataset = LineDataset(image_dir, label_file, processor)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    model = build_model().to(device)

    # 🔥 optional speed boost
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    losses = []

    for epoch in range(5):
        print(f"\n🔥 EPOCH {epoch+1}")

        loss = train_one_epoch(model, loader, optimizer, device)
        losses.append(loss)

        print(f"✅ LOSS: {loss:.4f}")

        debug_predictions(model, processor, dataset, device)

    os.makedirs("models/ocr", exist_ok=True)
    torch.save(model.state_dict(), "models/ocr/model.pth")

    print("\n💾 MODEL SAVED")
    print("📉 LOSS TREND:", losses)


if __name__ == "__main__":
    main()