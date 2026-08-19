import json
import os
from pathlib import Path

from paddleocr import PaddleOCR


INPUT_IMAGE = r"input/img_1.jpg"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True
)

results = list(ocr.predict(INPUT_IMAGE))

all_results = []

for page in results:

    texts = page["rec_texts"]
    confidences = page["rec_scores"]
    polygons = page["dt_polys"]
    boxes = page["rec_boxes"]

    page_result = []

    print("=" * 80)

    for text, conf, poly, box in zip(
            texts,
            confidences,
            polygons,
            boxes):

        item = {
            "text": text,
            "confidence": float(conf),
            "polygon": poly.tolist(),
            "bbox": box.tolist()
        }

        page_result.append(item)

        print(f"{conf:.3f} | {text}")

    all_results.append(page_result)

# Save JSON
with open(
    OUTPUT_DIR / "ocr_output.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

# Save Markdown
with open(
    OUTPUT_DIR / "ocr_output.md",
    "w",
    encoding="utf-8"
) as f:

    for page in all_results:

        for row in page:
            f.write(row["text"] + "\n")

        f.write("\n")

# Save visualization
for page in results:
    page.save_to_img(str(OUTPUT_DIR))

# Save native PaddleOCR JSON
for page in results:
    page.save_to_json(str(OUTPUT_DIR))

print("\nFinished!")
print("JSON :", OUTPUT_DIR / "ocr_output.json")
print("MD   :", OUTPUT_DIR / "ocr_output.md")
print("IMG  :", OUTPUT_DIR)