import time
from pathlib import Path

from schemas.ocr_schema import (
    OCRDocument,
    OCRManifest,
)

from utils.json_utils import JSONSerializer


class ManifestGenerator:


    @staticmethod
    def create(
        document: OCRDocument,
        start_time: float,
        outputs: dict,
    ) -> OCRManifest:


        total_words = 0
        confidence_sum = 0
        low_confidence = 0


        for page in document.pages:

            total_words += len(page.words)

            for word in page.words:

                confidence_sum += word.confidence

                if word.confidence_level == "LOW":

                    low_confidence += 1


        average_confidence = 0

        if total_words:

            average_confidence = (
                confidence_sum / total_words
            )


        return OCRManifest(

            file_name=document.file_name,

            ocr_engine="PaddleOCR 3.x",

            processing_time_seconds=round(
                time.time()-start_time,
                3
            ),

            total_pages=len(document.pages),

            total_words=total_words,

            average_confidence=round(
                average_confidence,
                3
            ),

            low_confidence_words=low_confidence,

            outputs=outputs,
        )



    @staticmethod
    def save(
        manifest: OCRManifest,
        path: str,
    ):

        JSONSerializer.save(
            manifest,
            path
        )