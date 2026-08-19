from pathlib import Path

from schemas.ocr_schema import OCRDocument,BlockType


class MarkdownExporter:
    """
    Converts OCRDocument into Markdown format.
    """

    @staticmethod
    def generate(document: OCRDocument) -> str:

        lines = []

        lines.append(f"# {document.file_name}")
        lines.append("")

        for page in document.pages:

            lines.append(f"<!-- Page {page.page_number} -->")
            lines.append("")

            for block in page.blocks:

                if block.is_title:

                    lines.append(f"# {block.text}")

                elif block.is_header:

                    lines.append(f"## {block.text}")

                elif block.is_text:

                    lines.append(block.text)

                elif block.is_table:

                    lines.append("> [TABLE DETECTED]")
                    lines.append(block.text)

                elif block.is_diagram:

                    lines.append("> [DIAGRAM DETECTED]")

                else:

                    lines.append(block.text)

                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save(
        document: OCRDocument,
        output_path: str | Path,
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        markdown = MarkdownExporter.generate(
            document
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(markdown)