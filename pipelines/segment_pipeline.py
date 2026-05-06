import os
from src.segmentation.segment_lines import segment_lines

input_dir = "data/processed"
output_dir = "data/lines"

for file in os.listdir(input_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, file)
        segment_lines(path, output_dir)