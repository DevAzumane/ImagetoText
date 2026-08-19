from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"

INTERMEDIATE_DIR = DATA_DIR / "intermediate"

OUTPUT_DIR = DATA_DIR / "output"

LOG_DIR = PROJECT_ROOT / "logs"

# Create folders automatically
for directory in [
    INPUT_DIR,
    INTERMEDIATE_DIR,
    OUTPUT_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# PaddleOCR Configuration
# ============================================================

OCR_LANGUAGE = "en"

USE_DOC_ORIENTATION = True

USE_DOC_UNWARPING = True

USE_TEXTLINE_ORIENTATION = True


# ============================================================
# OCR Processing Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.60

LOW_CONFIDENCE_THRESHOLD = 0.80

LINE_VERTICAL_TOLERANCE = 0.5

PARAGRAPH_VERTICAL_GAP_FACTOR = 1.5

PARAGRAPH_LEFT_ALIGNMENT_TOLERANCE = 20

TITLE_MAX_WORDS = 8

HEADER_MAX_WORDS = 6

TITLE_HEIGHT_FACTOR = 1.4

HEADER_HEIGHT_FACTOR = 1.15

CROP_PADDING = 10

# ============================================================
# Supported Image Types
# ============================================================

SUPPORTED_IMAGE_TYPES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
]

# ============================================================
# Output File Names
# ============================================================

OCR_JSON_NAME = "ocr_output.json"

OCR_MARKDOWN_NAME = "ocr_output.md"

OCR_DEBUG_IMAGE_NAME = "ocr_debug.png"

OCR_MANIFEST_NAME = "ocr_manifest.json"

# ============================================================
# Logging
# ============================================================

LOG_FILE = LOG_DIR / "ocr_pipeline.log"

LOG_LEVEL = "INFO"

# ==========================================================
# Qwen Configuration
# ==========================================================

LLAMA_SERVER_URL = "http://127.0.0.1:8080"

QWEN_MODEL_NAME = (
    r"..\qwen_model\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
)

QWEN_TEMPERATURE = 0.1

QWEN_MAX_TOKENS = 512

QWEN_IMAGE_SIZE = 1024