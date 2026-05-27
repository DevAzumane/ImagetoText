import re
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from configs.settings import (
    RUNS_DIR,
    GLYPH_OUTPUT_DIR,
    PAGE_TEMPLATES,
    FONT_OPTIONS,
)

from segmentation.segment_characters import CharacterSegmenter
from writer.handwriting_renderer import HandwritingRenderer
from pipeline.batch_pipeline import BatchPipeline

from engine.vlm_engine import QwenVLEngine

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

OUTPUT_FORMATS = ("png", "pdf", "zip")

HANDWRITING_MODES = ("custom", "font")


def _slug(value, fallback="notes"):
    slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")
    return slug or fallback

def _save_uploaded_images(files, destination):
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, file in enumerate(files, start=1):
        if not file or not getattr(file, "filename", ""):
            continue

        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {file.filename}")

        path = destination / f"img_{index}{suffix}"
        file.save(path)
        saved_paths.append(path)

    if not saved_paths:
        raise ValueError("Upload at least one input image.")

    return saved_paths

def _save_single_image(file, destination, filename):
    if not file or not getattr(file, "filename", ""):
        raise ValueError("Upload the handwriting alphabets image.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {file.filename}")

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{filename}{suffix}"
    file.save(path)
    return path

def _make_pdf(page_paths, output_path):
    images = [Image.open(path).convert("RGB") for path in page_paths]
    if not images:
        raise ValueError("No rendered pages were created.")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        resolution=100.0,
    )
    return output_path


def _make_zip(page_paths, summary_dir, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in page_paths:
            archive.write(path, arcname=f"pages/{path.name}")

        for path in sorted(summary_dir.glob("*.txt")):
            archive.write(path, arcname=f"summaries/{path.name}")

    return output_path

def run_automatic_job(
        run_id: str,
        input_files: list,
        alphabet_file: any,
        title: str,
        page_template: str = "classic",
        output_format: str = "pdf",
        handwriting_mode: str = "custom",
        font_key: str = "caveat",
        summary_lines: int = 3,
        progress_callback: callable = None,
    ):

    if page_template not in PAGE_TEMPLATES:
        raise ValueError(f"Unknown page format: {page_template}")

    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unknown output format: {output_format}")

    if handwriting_mode not in HANDWRITING_MODES:
        raise ValueError(f"Unknown handwriting option: {handwriting_mode}")

    if font_key not in FONT_OPTIONS:
        raise ValueError(f"Unknown font: {font_key}")

    # run_id = uuid.uuid4().hex[:10]
    run_dir = RUNS_DIR / run_id
    upload_dir = run_dir / "uploads"
    input_dir = upload_dir / "input"
    ocr_output_dir = run_dir / "summaries"
    render_dir = run_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(5, "Saving uploaded images")

    _save_uploaded_images(input_files, input_dir)

    if handwriting_mode == "custom":
        if progress_callback:
            progress_callback(10, "Preparing user handwriting")

        alphabet_path = _save_single_image(
            alphabet_file,
            upload_dir,
            "handwriting_alphabets",
        )

        CharacterSegmenter(clean_output=True).segment(
            str(alphabet_path),
            str(GLYPH_OUTPUT_DIR),
        )
    elif progress_callback:
        progress_callback(15, "Using selected preexisting font")

    if progress_callback:
        progress_callback(20, "Starting OCR and summaries")

    def pipeline_progress(completed_steps, total_steps, message):
        if not progress_callback:
            return
        percent = 20 + int((completed_steps / max(1, total_steps)) * 50)
        progress_callback(percent, message)

    BatchPipeline().process_folder(
        input_folder=str(input_dir),
        output_folder=str(ocr_output_dir),
        summary_lines=int(summary_lines),
        progress_callback=pipeline_progress,
    )

    if progress_callback:
        progress_callback(75, "Rendering handwritten pages")

    output_stem = _slug(title)
    renderer = HandwritingRenderer(
        font_path=str(FONT_OPTIONS[font_key]["path"])
    )
    page_paths = renderer.render_summaries(
        input_dir=str(ocr_output_dir),
        output_path=str(render_dir / f"{output_stem}.png"),
        template=page_template,
        handwriting_source="glyphs" if handwriting_mode == "custom" else "font",
        custom_title=title.strip() or "Notes",
    )

    if progress_callback:
        progress_callback(90, "Packaging output")

    if output_format == "pdf":
        download_path = _make_pdf(
            page_paths,
            render_dir / f"{output_stem}.pdf",
        )
    elif output_format == "zip" or len(page_paths) > 1:
        download_path = _make_zip(
            page_paths,
            ocr_output_dir,
            render_dir / f"{output_stem}.zip",
        )
    else:
        download_path = page_paths[0]

    return {
        "run_id": run_id,
        "download_path": download_path,
        "page_paths": page_paths,
        "summary_dir": ocr_output_dir,
    }
