from pathlib import Path

from services.ocr_service import OCRService
from utils.logger import get_logger


logger = get_logger("APP")


def main():

    # Project root
    base_dir = Path(__file__).resolve().parent

    input_dir = (
        base_dir
        / "data"
        / "input"
    )


    # Supported image formats

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }


    images = [
        img
        for img in input_dir.iterdir()
        if img.suffix.lower() in image_extensions
    ]


    if not images:

        print(
            "No images found in input folder"
        )

        return


    logger.info(
        f"Found {len(images)} images"
    )


    # Load PaddleOCR once

    ocr_service = OCRService()


    for index, image_path in enumerate(images, start=1):

        try:

            print(
                f"\nProcessing {index}/{len(images)}"
            )

            print(
                f"Image: {image_path.name}"
            )


            document = ocr_service.run(
                str(image_path)
            )


            total_words = sum(
                len(page.words)
                for page in document.pages
            )


            print(
                f"Completed: {image_path.name}"
            )

            print(
                f"Words detected: {total_words}"
            )


        except Exception as e:

            logger.exception(
                f"Failed processing {image_path.name}"
            )


    print(
        "\nAll images processed"
    )


if __name__ == "__main__":

    main()