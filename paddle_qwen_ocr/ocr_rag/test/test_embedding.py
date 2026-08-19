# import time
# from models.embedding.embedding_model import EmbeddingModel

# start = time.time()
# model = EmbeddingModel()
# print(f"Model Load: {time.time() - start:.2f}s")

# start = time.time()
# embedding = model.encode("Time Series Forecasting")
# print(f"First Encode: {time.time() - start:.2f}s")

# start = time.time()
# embedding = model.encode("Trend is a long term movement.")
# print(f"Second Encode: {time.time() - start:.2f}s")

# from services.embedding_service import EmbeddingService

# service = EmbeddingService()

# embedding = service.generate(
#     "Time Series Forecasting"
# )

# print(len(embedding))
# print(embedding[:10])

from services.ocr_service import OCRService
from pipelines.chunk_pipeline import ChunkPipeline
from pipelines.embedding_pipeline import EmbeddingPipeline


ocr = OCRService()

document = ocr.run(
    "data/input/handwritten_01.jpg"
)

chunks = ChunkPipeline().run(document)

chunks = EmbeddingPipeline().run(chunks)

print("=" * 80)

print(f"Total Chunks : {len(chunks)}")

print("=" * 80)

for chunk in chunks:

    print(chunk.chunk_id)

    print(chunk.chunk_type)

    print(chunk.title)

    print(f"Embedding Dimension : {len(chunk.embedding)}")

    print(chunk.embedding[:5])

    print("-" * 60)