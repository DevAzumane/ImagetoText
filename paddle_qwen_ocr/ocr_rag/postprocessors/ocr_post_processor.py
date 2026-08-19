import re
import unicodedata
from utils.logger import get_logger
from copy import deepcopy

from config import (
    CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LINE_VERTICAL_TOLERANCE,
    PARAGRAPH_VERTICAL_GAP_FACTOR,
    PARAGRAPH_LEFT_ALIGNMENT_TOLERANCE,
    TITLE_MAX_WORDS,
    HEADER_MAX_WORDS,
    TITLE_HEIGHT_FACTOR,
    HEADER_HEIGHT_FACTOR
)

from schemas.ocr_schema import (
    OCRDocument,
    OCRBlock,
    BoundingBox,
    BlockType,
)

logger = get_logger(__name__)

class OCRPostProcessor:

    @classmethod
    def process(cls, document: OCRDocument) -> OCRDocument:

        cls._remove_empty_words(document)

        cls._normalize_text(document)

        cls._normalize_unicode(document)

        cls._sort_reading_order(document)

        cls._assign_reading_order(document)

        cls._assign_confidence(document)

        cls._build_blocks(document)

        cls._merge_paragraphs(document)

        cls._classify_layout(document)

        cls._select_qwen_candidates(document)

        cls._calculate_statistics(document)

        return document

    # ====================================================
    # Remove Empty OCR Words
    # ====================================================

    @staticmethod
    def _remove_empty_words(document):

        for page in document.pages:

            page.words = [
                word
                for word in page.words
                if word.text and word.text.strip()
            ]

    # ====================================================
    # Normalize Spaces
    # ====================================================

    @staticmethod
    def _normalize_text(document):

        for page in document.pages:

            for word in page.words:

                word.text = re.sub(
                    r"\s+",
                    " ",
                    word.text.strip()
                )

    # ====================================================
    # Normalize Unicode
    # ====================================================

    @staticmethod
    def _normalize_unicode(document):

        for page in document.pages:

            for word in page.words:

                word.text = unicodedata.normalize(
                    "NFKC",
                    word.text
                )

    # ====================================================
    # Reading Order
    # ====================================================

    @staticmethod
    def _sort_reading_order(document):

        for page in document.pages:

            page.words.sort(
                key=lambda w: (
                    w.bbox.top,
                    w.bbox.left
                )
            )

    # ====================================================
    # Reading Order Index
    # ====================================================

    @staticmethod
    def _assign_reading_order(document):

        for page in document.pages:

            for idx, word in enumerate(
                page.words,
                start=1
            ):

                word.reading_order = idx

    # ====================================================
    # Confidence Labels
    # ====================================================

    @staticmethod
    def _assign_confidence(document):

        for page in document.pages:

            for word in page.words:

                if word.confidence >= LOW_CONFIDENCE_THRESHOLD:

                    word.confidence_level = "HIGH"

                elif word.confidence >= CONFIDENCE_THRESHOLD:

                    word.confidence_level = "MEDIUM"

                else:

                    word.confidence_level = "LOW"

    # ====================================================
    # Build Blocks
    # ====================================================                

    @staticmethod
    def _build_blocks(document):

        block_id = 1

        for page in document.pages:

            page.blocks = []

            page.blocks.clear()

            if not page.words:
                continue

            current_words = [page.words[0]]

            for word in page.words[1:]:

                previous = current_words[-1]

                avg_height = (
                    previous.bbox.height +
                    word.bbox.height
                ) / 2

                if abs(word.bbox.top - previous.bbox.top) <= avg_height * LINE_VERTICAL_TOLERANCE:

                    current_words.append(word)

                else:

                    OCRPostProcessor._create_block(
                        page,
                        current_words,
                        block_id
                    )

                    block_id += 1

                    current_words = [word]

            OCRPostProcessor._create_block(
                page,
                current_words,
                block_id
            )

            block_id += 1

            page.total_blocks = len(page.blocks)

            for block in page.blocks:
                logger.info(
                    f"Block {block.id}: {block.text}"
                )


    @staticmethod
    def _create_block(page, words, block_id):

        left = min(w.bbox.left for w in words)
        top = min(w.bbox.top for w in words)
        right = max(w.bbox.right for w in words)
        bottom = max(w.bbox.bottom for w in words)

        bbox = BoundingBox(
            points=[
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom]
            ]
        )

        text = " ".join(
            word.text
            for word in words
        )

        confidence = sum(
            word.confidence
            for word in words
        ) / len(words)

        word_ids=[word.id for word in words]

        block = OCRBlock(
            id=block_id,
            text=text,
            bbox=bbox,
            word_ids=word_ids,
            confidence=confidence,
            block_type=BlockType.TEXT,
            line_count=1,
            page_number=page.page_number,
            word_count=len(words),
            first_word_id=word_ids[0],
            last_word_id=word_ids[-1]
        )

        page.blocks.append(block)

        for word in words:
            word.block_id = block.id                 


    # ====================================================
    # Merge Blocks
    # ====================================================   

    @staticmethod
    def _merge_paragraphs(document):

        for page in document.pages:

            if len(page.blocks) <= 1:
                continue

            merged_blocks = []

            current = deepcopy(page.blocks[0])

            for nxt in page.blocks[1:]:

                if OCRPostProcessor._should_merge(current, nxt):

                    OCRPostProcessor._merge_two_blocks(
                        current,
                        nxt
                    )

                else:

                    merged_blocks.append(current)

                    current = deepcopy(nxt)

            merged_blocks.append(current)

            page.blocks = merged_blocks

            page.total_blocks = len(page.blocks)   

            logger.info(
                f"Paragraph merging completed. "
                f"Total blocks: {page.total_blocks}"
            )      


    @staticmethod
    def _should_merge(current, nxt):

        if current.page_number != nxt.page_number:
            return False

        if current.block_type != nxt.block_type:
            return False

        vertical_gap = (
            nxt.bbox.top
            - current.bbox.bottom
        )

        average_height = (
            current.bbox.height +
            nxt.bbox.height
        ) / 2

        if vertical_gap > average_height * PARAGRAPH_VERTICAL_GAP_FACTOR:
            return False

        left_difference = abs(
            current.bbox.left
            - nxt.bbox.left
        )

        if left_difference > PARAGRAPH_LEFT_ALIGNMENT_TOLERANCE:
            return False

        return True             


    @staticmethod
    def _merge_two_blocks(current, nxt):

        current.text += "\n" + nxt.text

        current.word_ids.extend(
            nxt.word_ids
        )

        current.word_count += nxt.word_count

        current.line_count += nxt.line_count

        current.last_word_id = nxt.last_word_id

        current.confidence = (
            current.confidence +
            nxt.confidence
        ) / 2

        left = min(
            current.bbox.left,
            nxt.bbox.left
        )

        top = min(
            current.bbox.top,
            nxt.bbox.top
        )

        right = max(
            current.bbox.right,
            nxt.bbox.right
        )

        bottom = max(
            current.bbox.bottom,
            nxt.bbox.bottom
        )

        current.bbox.points = [

            [left, top],

            [right, top],

            [right, bottom],

            [left, bottom]
        ]


    @staticmethod
    def _classify_layout(document):

        for page in document.pages:

            if not page.blocks:
                continue

            average_height = (
                sum(
                    block.bbox.height
                    for block in page.blocks
                )
                / len(page.blocks)
            )

            for index, block in enumerate(page.blocks):

                OCRPostProcessor._classify_block(
                    block,
                    average_height,
                    index
                )
            for block in page.blocks:

                logger.info(
                    f"Block {block.id} -> {block.block_type}"
                )

    @staticmethod
    def _classify_block(
        block,
        average_height,
        index
    ):

        word_count = block.word_count

        # -------------------------------
        # TITLE
        # -------------------------------

        if (
            index == 0
            and word_count <= TITLE_MAX_WORDS
            and block.bbox.height >= average_height * TITLE_HEIGHT_FACTOR
        ):

            block.block_type = BlockType.TITLE
            return

        # -------------------------------
        # HEADER
        # -------------------------------

        if (
            word_count <= HEADER_MAX_WORDS
            and block.bbox.height >= average_height * HEADER_HEIGHT_FACTOR
        ):

            block.block_type = BlockType.HEADER
            return

        # -------------------------------
        # Default
        # -------------------------------

        block.block_type = BlockType.TEXT



    # ====================================================
    # Qwen Selection
    # ====================================================

    @staticmethod
    def _select_qwen_candidates(document):

        for page in document.pages:

            for block in page.blocks:

                if block.block_type == BlockType.TABLE:

                    block.requires_qwen = True
                    block.qwen_reason = "table"

                elif block.block_type == BlockType.DIAGRAM:

                    block.requires_qwen = True
                    block.qwen_reason = "diagram"

                elif block.confidence < LOW_CONFIDENCE_THRESHOLD:

                    block.requires_qwen = True
                    block.qwen_reason = "low_confidence"

                else:

                    block.requires_qwen = False
                    block.qwen_reason = None

    # ====================================================
    # Statistics
    # ====================================================

    @staticmethod
    def _calculate_statistics(document):

        for page in document.pages:

            total_confidence = 0.0

            low_confidence = 0

            for word in page.words:

                total_confidence += word.confidence

                if word.confidence_level == "LOW":

                    low_confidence += 1

            page.total_words = len(page.words)

            page.low_confidence_count = low_confidence

            if page.total_words:

                page.average_confidence = (
                    total_confidence / page.total_words
                )
            else:

                page.average_confidence = 0.0
