from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from schemas.ocr_schema import BlockType
import tempfile

from PIL import Image, ImageOps

from config import SUPPORTED_IMAGE_TYPES, CROP_PADDING


@dataclass(slots=True)
class ImageMetadata:
    file_name: str
    file_path: str
    width: int
    height: int
    image_mode: str
    image_format: str


class ImageValidator:

    @staticmethod
    def validate(image_path: str | Path) -> Tuple[Image.Image, ImageMetadata]:

        image_path = Path(image_path)

        # --------------------------
        # File Exists
        # --------------------------

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found : {image_path}"
            )

        # --------------------------
        # Extension Check
        # --------------------------

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image type : {image_path.suffix}"
            )

        # --------------------------
        # Read Image
        # --------------------------

        try:

            image = Image.open(image_path)

        except Exception as e:

            raise RuntimeError(
                f"Unable to open image : {e}"
            )

        # --------------------------
        # Correct EXIF Rotation
        # --------------------------

        image = ImageOps.exif_transpose(image)

        # --------------------------
        # Convert to RGB
        # --------------------------

        if image.mode != "RGB":

            image = image.convert("RGB")

        metadata = ImageMetadata(
            file_name=image_path.name,
            file_path=str(image_path.resolve()),
            width=image.width,
            height=image.height,
            image_mode=image.mode,
            image_format=image.format,
        )

        return image, metadata

class ImageCropper:

    PADDING = CROP_PADDING

    @staticmethod
    def crop_regions(
        image_path: str | Path,
        document,
        output_dir: str | Path,
    ):

        image = Image.open(image_path).convert("RGB")

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        crop_count = 0

        for page in document.pages:

            for block in page.blocks:

                if block.block_type not in (
                    BlockType.TABLE,
                    BlockType.DIAGRAM,
                ):
                    continue

                left = max(
                    0,
                    int(block.bbox.left - ImageCropper.PADDING),
                )

                top = max(
                    0,
                    int(block.bbox.top - ImageCropper.PADDING),
                )

                right = min(
                    image.width,
                    int(block.bbox.right + ImageCropper.PADDING),
                )

                bottom = min(
                    image.height,
                    int(block.bbox.bottom + ImageCropper.PADDING),
                )

                crop = image.crop(
                    (
                        left,
                        top,
                        right,
                        bottom,
                    )
                )

                crop_name = (
                    f"page_{page.page_number}_"
                    f"{block.block_type}_"
                    f"{block.id}.png"
                )

                crop.save(
                    output_dir / crop_name
                )

                block.crop_path = str(
                    output_dir / crop_name
                )
                
                crop_count += 1

        return crop_count    

class ImageUtils:

    @staticmethod
    def resize_for_qwen(

        image_path: str,

        max_size: int = 1024,

    ) -> str:

        image = Image.open(image_path)

        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        temp = tempfile.NamedTemporaryFile(

            suffix=".jpg",

            delete=False,

        )

        image.save(

            temp.name,

            format="JPEG",

            quality=90,

        )

        return temp.name    