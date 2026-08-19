from pathlib import Path

from pipelines.paddle_pipeline import PaddleOCRPipeline
from postprocessors.ocr_post_processor import OCRPostProcessor
from services.qwen_service import QwenService


def main():

    image_path = Path(
        "data/input/handwritten_01.jpg"
    )

    # ---------------------------------------
    # Paddle OCR
    # ---------------------------------------

    print("=" * 80)
    print("Running PaddleOCR...")
    print("=" * 80)

    paddle = PaddleOCRPipeline()

    document = paddle.run(str(image_path))

    document = OCRPostProcessor.process(document)

    print("OCR Completed")
    print()

    # ---------------------------------------
    # Show OCR Blocks
    # ---------------------------------------

    print("=" * 80)
    print("OCR Blocks")
    print("=" * 80)

    for page in document.pages:

        print(f"\nPage {page.page_number}\n")

        for block in page.blocks:

            print(f"Block {block.id}")
            print(f"Type       : {block.block_type}")
            print(f"Confidence : {block.confidence:.2f}")
            print(block.text)
            print("-" * 60)

    # ---------------------------------------
    # Qwen Enhancement
    # ---------------------------------------

    print("\n")
    print("=" * 80)
    print("Sending to Qwen...")
    print("=" * 80)

    qwen = QwenService()

    response = qwen.analyze(
        image_path=str(image_path),
        document=document,
    )

    print("\n")
    print("=" * 80)
    print("Qwen Response")
    print("=" * 80)

    print(response)


if __name__ == "__main__":
    main()