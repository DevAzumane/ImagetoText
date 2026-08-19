from pathlib import Path
import time

from pipelines.paddle_pipeline import PaddleOCRPipeline
from postprocessors.ocr_post_processor import OCRPostProcessor

from utils.image_utils import ImageValidator, ImageCropper
from utils.logger import get_logger
from utils.json_utils import JSONSerializer
from utils.markdown_utils import MarkdownExporter
from utils.visualization import OCRVisualizer
from utils.manifest import ManifestGenerator
from pipelines.qwen_pipeline import QwenPipeline
from pipelines.chunk_pipeline import ChunkPipeline
from utils.chunk_serializer import ChunkSerializer
from pipelines.embedding_pipeline import EmbeddingPipeline

logger = get_logger(__name__)


class OCRService:

    def __init__(self):

        logger.info("Initializing OCR Service")

        self.pipeline = PaddleOCRPipeline()
        self.qwen_pipeline = QwenPipeline()

        logger.info("OCR Service Ready")

    def run(self, image_path: str):

        start_time = time.time()

        image_name = Path(image_path).stem

        output_dir = (
            Path("data")
            / "output"
            / image_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info(
            f"Starting OCR pipeline for {image_path}"
        )

        # -------------------------
        # 1. Validate Image
        # -------------------------

        image, metadata = ImageValidator.validate(
            image_path
        )

        # -------------------------
        # 2. Run PaddleOCR
        # -------------------------

        document = self.pipeline.run(
            image_path
        )

        # -------------------------
        # 3. Post Processing
        # -------------------------

        document = OCRPostProcessor.process(
            document
        )

        document = self.qwen_pipeline.run(
            image_path=image_path,
            document=document,
        )


        # -------------------------
        # Chunking
        # -------------------------
        chunks = ChunkPipeline().run(document)

        chunks = EmbeddingPipeline().run(chunks)

        chunk_path = (
            output_dir
            / "chunks.json"
        )

        ChunkSerializer.save(
            chunks,
            chunk_path,
        )

        ChunkSerializer.save(
            chunks,
            output_dir / "embedded_chunks.json"
        )

        # -------------------------
        # 4. Save JSON
        # -------------------------

        json_path = (
            output_dir
            / "ocr_document.json"
        )

        JSONSerializer.save(
            document,
            json_path
        )

        # -------------------------
        # 5. Save Markdown
        # -------------------------

        markdown_path = (
            output_dir
            / "document.md"
        )

        MarkdownExporter.save(
            document,
            markdown_path
        )
        enhanced_path = (
            output_dir
            / "qwen_document.md"
        )

        with open(
            enhanced_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(document.enhanced_markdown)

        crop_dir = output_dir / "crops"

        ImageCropper.crop_regions(
            image_path=image_path,
            document=document,
            output_dir=crop_dir,
        )

        # -------------------------
        # 6. Save Debug Image
        # -------------------------

        debug_path = (
            output_dir
            / "debug_ocr.png"
        )

        OCRVisualizer.draw(
            image_path,
            document,
            str(debug_path)
        )

        # -------------------------
        # 7. Manifest
        # -------------------------

        manifest = ManifestGenerator.create(
            document,
            start_time,
            outputs={
                "json": str(json_path),
                "markdown": str(markdown_path),
                "enhanced_markdown": str(enhanced_path),
                "debug": str(debug_path),
                "chunks": str(chunk_path),
            }
        )

        ManifestGenerator.save(
            manifest,
            output_dir / "manifest.json"
        )

        logger.info(
            "OCR pipeline completed successfully"
        )

        return document