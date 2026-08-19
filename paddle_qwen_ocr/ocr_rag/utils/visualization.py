from pathlib import Path

import cv2

from schemas.ocr_schema import OCRDocument


class OCRVisualizer:
    """
    Creates OCR debugging images
    with bounding boxes and confidence.
    """

    @staticmethod
    def draw(
        image_path: str,
        document: OCRDocument,
        output_path: str,
    ):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Unable to read image {image_path}"
            )


        for page in document.pages:

            for word in page.words:

                points = word.bbox.points


                # Convert bbox points

                x1 = points[0][0]
                y1 = points[0][1]

                x2 = points[2][0]
                y2 = points[2][1]


                # Draw rectangle

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2,
                )


                label = (
                    f"{word.text}"
                    f" ({word.confidence:.2f})"
                )


                cv2.putText(
                    image,
                    label,
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    1,
                )


        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        cv2.imwrite(
            str(output_path),
            image
        )