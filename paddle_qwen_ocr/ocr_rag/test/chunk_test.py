from pathlib import Path

from pipelines.chunk_pipeline import ChunkPipeline
from schemas.ocr_schema import OCRDocument

document = OCRDocument(
    file_name="handwritten_01.jpg"
)

document.enhanced_markdown = Path(
    "data/output/handwritten_01/qwen_document.md"
).read_text(
    encoding="utf-8"
)

chunks = ChunkPipeline().run(document)

print("=" * 80)
print(f"Total Chunks : {len(chunks)}")
print("=" * 80)

for chunk in chunks:

    print(chunk.chunk_id)
    print(chunk.chunk_type)
    print(chunk.title)
    print("-" * 60)
    print(chunk.text)
    print("=" * 80)