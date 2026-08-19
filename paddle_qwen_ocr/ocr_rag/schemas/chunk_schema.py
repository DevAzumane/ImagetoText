from enum import Enum
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ChunkType(str, Enum):
    DOCUMENT = "document"
    PAGE = "page"
    SECTION = "section"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    LIST_ITEM = "list_item"
    DEFINITION = "definition"


@dataclass(slots=True)
class DocumentChunk:

    chunk_id: str

    chunk_type: ChunkType

    page_number: int

    title: str

    text: str

    parent_title: str | None = None

    keywords: List[str] = field(default_factory=list)

    entities: List[str] = field(default_factory=list)

    relationships: List[Dict] = field(default_factory=list)

    category: str = ""

    embedding: List[float] = field(default_factory=list)