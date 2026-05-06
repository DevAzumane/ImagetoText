import torch
from PIL import Image
from difflib import SequenceMatcher


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def validate_labels(model, processor, dataset, device, threshold=0.5):
    model.eval()

    bad_samples = []

    for idx in range(len(dataset)):
        img_name = dataset.images[idx]
        gt_text = dataset.texts[idx]

        img_path = f"{dataset.image_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            output_ids = model.generate(pixel_values)

        pred_text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        score = similarity(gt_text.lower(), pred_text.lower())

        print(f"\n🖼 {img_name}")
        print(f"GT  : {gt_text}")
        print(f"PRED: {pred_text}")
        print(f"SIM : {score:.2f}")

        if score < threshold:
            bad_samples.append((img_name, gt_text, pred_text, score))

    return bad_samples