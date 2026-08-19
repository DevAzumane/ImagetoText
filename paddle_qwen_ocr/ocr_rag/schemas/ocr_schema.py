from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ==========================================================
# Enums
# ==========================================================

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BlockType(str, Enum):

    TEXT = "text"
    TITLE = "title"
    HEADER = "header"
    TABLE = "table"
    DIAGRAM = "diagram"
    FOOTER = "footer"
    LIST = "list"


# ==========================================================
# Bounding Box
# ==========================================================

@dataclass(slots=True)
class BoundingBox:
    """
    Represents a quadrilateral bounding box.
    """

    points: List[List[int]]

    @property
    def left(self):
        return min(point[0] for point in self.points)

    @property
    def right(self):
        return max(point[0] for point in self.points)

    @property
    def top(self):
        return min(point[1] for point in self.points)

    @property
    def bottom(self):
        return max(point[1] for point in self.points)

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def center_x(self):
        return self.left + self.width / 2

    @property
    def center_y(self):
        return self.top + self.height / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        if self.height == 0:
            return 0.0
        return self.width / self.height


# ==========================================================
# OCR Word
# ==========================================================

@dataclass(slots=True)
class OCRWord:

    @property
    def left(self):
        return self.bbox.left

    @property
    def right(self):
        return self.bbox.right

    @property
    def top(self):
        return self.bbox.top

    @property
    def bottom(self):
        return self.bbox.bottom

    id: int

    text: str

    confidence: float

    bbox: BoundingBox

    confidence_level: Optional[ConfidenceLevel] = None

    line_number: Optional[int] = None

    reading_order: Optional[int] = None

    normalized_text: Optional[str] = None

    block_id: Optional[int] = None

    page_id: Optional[int] = None

    is_header: bool = False

    is_table: bool = False

    is_diagram: bool = False


# ==========================================================
# OCR Block
# ==========================================================

@dataclass(slots=True)
class OCRBlock:

    id: int

    text: str

    bbox: BoundingBox

    word_ids: List[int] = field(default_factory=list)

    block_type: BlockType = BlockType.TEXT

    confidence: float = 0.0

    line_count: int = 1

    page_number: int = 0

    word_count: int = 0

    first_word_id: int = 0

    last_word_id: int = 0

    requires_qwen: bool = False

    crop_path: Optional[str] = None

    enhanced_text: Optional[str] = None

    qwen_reason: Optional[str] = None

    @property
    def is_title(self):
        return self.block_type == BlockType.TITLE

    @property
    def is_header(self):
        return self.block_type == BlockType.HEADER

    @property
    def is_table(self):
        return self.block_type == BlockType.TABLE

    @property
    def is_diagram(self):
        return self.block_type == BlockType.DIAGRAM

    @property
    def is_text(self):
        return self.block_type == BlockType.TEXT


# ==========================================================
# OCR Page
# ==========================================================

@dataclass(slots=True)
class OCRPage:

    page_number: int

    words: List[OCRWord] = field(default_factory=list)

    blocks: List[OCRBlock] = field(default_factory=list)

    average_confidence: float = 0.0

    low_confidence_count: int = 0

    total_words: int = 0

    total_blocks: int = 0

    def add_word(self, word: OCRWord):
        self.words.append(word)

    def add_block(self, block: OCRBlock):
        self.blocks.append(block)


# ==========================================================
# OCR Document
# ==========================================================

@dataclass(slots=True)
class OCRDocument:
    """
    Complete OCR document.
    """

    file_name: str

    version: str = "1.0"

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    pages: List[OCRPage] = field(default_factory=list)

    markdown: str = ""

    enhanced_markdown: str = ""

    metadata: Dict = field(default_factory=dict)

    statistics: Dict = field(default_factory=dict)

    def add_page(self, page: OCRPage):
        self.pages.append(page)


# ==========================================================
# OCR Manifest
# ==========================================================

@dataclass(slots=True)
class OCRManifest:

    file_name: str

    ocr_engine: str

    processing_time_seconds: float

    total_pages: int

    total_words: int

    average_confidence: float

    low_confidence_words: int

    version: str = "1.0"

    statistics: Dict = field(default_factory=dict)

    outputs: Dict[str, str] = field(default_factory=dict)