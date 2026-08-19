from paddleocr import PaddleOCR

from config import (
    OCR_LANGUAGE,
    USE_DOC_ORIENTATION,
    USE_DOC_UNWARPING,
    USE_TEXTLINE_ORIENTATION,
)

from schemas.ocr_schema import (
    BoundingBox,
    OCRDocument,
    OCRPage,
    OCRWord,
)

from utils.logger import get_logger

from postprocessors.ocr_post_processor import OCRPostProcessor

logger = get_logger(__name__)


class PaddleOCRPipeline:

    def __init__(self):

        logger.info("Initializing PaddleOCR...")

        self.ocr = PaddleOCR(
            lang=OCR_LANGUAGE,
            use_doc_orientation_classify=USE_DOC_ORIENTATION,
            use_doc_unwarping=USE_DOC_UNWARPING,
            use_textline_orientation=USE_TEXTLINE_ORIENTATION,
        )

        logger.info("PaddleOCR Initialized Successfully.")

    def run(
        self,
        image_path: str,
    ) -> OCRDocument:

        logger.info(f"Running OCR on {image_path}")

        result = self.ocr.predict(image_path)

        document = OCRDocument(
            file_name=image_path.split("\\")[-1]
        )

        page = OCRPage(page_number=1)

        word_id = 1

        for prediction in result:

            boxes = prediction["rec_boxes"]

            texts = prediction["rec_texts"]

            scores = prediction["rec_scores"]

            for box, text, score in zip(
                boxes,
                texts,
                scores,
            ):

                bbox = BoundingBox(
                    points=[
                        [int(box[0]), int(box[1])],
                        [int(box[2]), int(box[1])],
                        [int(box[2]), int(box[3])],
                        [int(box[0]), int(box[3])],
                    ]
                )

                word = OCRWord(
                    id=word_id,
                    text=text,
                    confidence=float(score),
                    bbox=bbox,
                )

                page.add_word(word)

                word_id += 1

        document.add_page(page)

        logger.info(
            f"OCR Completed. "
            f"{len(page.words)} text regions detected."
        )

        document = OCRPostProcessor.process(document)

        return document