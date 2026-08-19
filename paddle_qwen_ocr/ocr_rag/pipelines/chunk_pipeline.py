import re

from schemas.chunk_schema import (
    DocumentChunk,
    ChunkType,
)


class ChunkPipeline:

    def run(self, document):

        markdown = document.enhanced_markdown

        chunks = []

        chunks.extend(
            self._document_chunk(document)
        )

        chunks.extend(
            self._title_chunk(markdown)
        )

        chunks.extend(
            self._section_chunks(markdown)
        )

        chunks.extend(
            self._list_chunks(markdown)
        )

        chunks.extend(
            self._formula_chunks(markdown)
        )

        chunks.extend(
            self._definition_chunks(markdown)
        )

        return chunks

    # =====================================================
    # Document Chunk
    # =====================================================

    def _document_chunk(self, document):

        title = self._extract_title(
            document.enhanced_markdown
        )

        return [

            DocumentChunk(

                chunk_id="document",

                chunk_type=ChunkType.DOCUMENT,

                page_number=1,

                title=title,

                text=document.enhanced_markdown,

            )

        ]

    # =====================================================
    # Section Chunks
    # =====================================================

    def _section_chunks(self, markdown):

        chunks = []

        pattern = r"^(?:##\s*)?([A-Za-z0-9 ()-]+):?\s*$"

        matches = list(
            re.finditer(
                pattern,
                markdown,
                flags=re.MULTILINE,
            )
        )

        for idx, match in enumerate(matches):

            title = match.group(1).strip()

            start = match.end()

            if idx == len(matches) - 1:
                end = len(markdown)
            else:
                end = matches[idx + 1].start()

            text = markdown[start:end]

            filtered = []

            for line in text.splitlines():

                if "=" in line:
                    continue

                filtered.append(line)

            text = "\n".join(filtered).strip()

            chunks.append(

                DocumentChunk(
                    chunk_id=f"section_{idx+1}",
                    chunk_type=ChunkType.SECTION,
                    page_number=1,
                    title=title,
                    parent_title=self._extract_title(markdown),
                    text=text,
                )

            )

        return chunks

    # =====================================================
    # Formula Chunks
    # =====================================================

    def _formula_chunks(self, markdown):

        chunks = []

        parent = None

        for line in markdown.splitlines():

            line = line.strip()

            if (
                line.endswith(":")
                and not line.startswith("#")
            ):

                parent = line[:-1]

            if "=" not in line:
                continue

            chunks.append(

                DocumentChunk(

                    chunk_id=f"formula_{len(chunks)+1}",

                    chunk_type=ChunkType.FORMULA,

                    page_number=1,

                    title="Formula",

                    parent_title=parent,

                    text=line,

                )

            )

        return chunks

    # =====================================================
    # List Chunks
    # =====================================================

    def _list_chunks(self, markdown):

        chunks = []

        parent = None

        lines = markdown.splitlines()

        for line in lines:

            line = line.strip()

            if (
                line.endswith(":")
                and not line.startswith("#")
            ):

                parent = line[:-1]

            match = re.match(
                r"^(\d+)\.\s+(.*)",
                line,
            )

            if not match:
                continue

            text = match.group(2).strip()

            title = text.split("-")[0].strip()

            chunks.append(

                DocumentChunk(

                    chunk_id=f"list_{len(chunks)+1}",

                    chunk_type=ChunkType.LIST_ITEM,

                    page_number=1,

                    title=title,

                    parent_title=parent,

                    text=text,

                )

            )

        return chunks

    # =====================================================
    # Definition Chunks
    # =====================================================

    def _definition_chunks(self, markdown):

        chunks = []

        pattern = r"([A-Za-z ()]+):\n- (.+)"

        matches = re.findall(
            pattern,
            markdown,
            flags=re.MULTILINE,
        )

        for idx, (title, definition) in enumerate(matches):

            chunks.append(

                DocumentChunk(

                    chunk_id=f"definition_{idx+1}",

                    chunk_type=ChunkType.DEFINITION,

                    page_number=1,

                    title=title.strip(),

                    text=definition.strip(),

                )

            )

        return chunks


    # =====================================================
    # Title Chunk
    # =====================================================

    def _title_chunk(self, markdown):

        title = self._extract_title(markdown)

        return [

            DocumentChunk(

                chunk_id="title",

                chunk_type=ChunkType.SECTION,

                page_number=1,

                title="Document Title",

                text=title,

            )

        ]

    # =====================================================
    # Helpers
    # =====================================================

    def _extract_title(self, markdown):

        match = re.search(
            r"^# (.+)$",
            markdown,
            flags=re.MULTILINE,
        )

        if match:

            return match.group(1).strip()

        return "Document"