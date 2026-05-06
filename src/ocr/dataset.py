import os
from PIL import Image
from torch.utils.data import Dataset


class LineDataset(Dataset):
    def __init__(self, image_dir, label_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" not in line:
                    continue

                img_name, text = line.split("|", 1)

                img_path = os.path.join(image_dir, img_name)

                if os.path.exists(img_path):
                    self.samples.append((img_name, text))
                else:
                    print(f"⚠️ Missing image: {img_name}")

        print(f"✅ Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text = self.samples[idx]

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values[0]

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).input_ids[0]

        # ignore padding in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels, text, img_name