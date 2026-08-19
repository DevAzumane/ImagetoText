from schemas.ocr_schema import OCRDocument


class PromptBuilder:

    @staticmethod
    def build(document: OCRDocument) -> str:

        prompt = []

        prompt.append("""
                You are an expert handwritten document parser.

                Use the image as the primary source and the OCR text only as guidance.

                Reconstruct the document exactly as it appears.

                Rules:
                - Preserve the original wording whenever readable.
                - Correct only obvious OCR recognition mistakes.
                - Do NOT rewrite, summarize, expand or explain.
                - Do NOT improve grammar or sentence structure.
                - Do NOT invent missing information.
                - Detect and preserve the document title.
                - Detect headings and subheadings.
                - Preserve numbered lists and bullet lists.
                - Preserve the reading order.
                - Preserve tables as Markdown tables whenever possible.
                - Preserve diagrams by adding a short Markdown description only if text cannot represent them.
                - Preserve formulas EXACTLY as written.
                - NEVER convert formulas into LaTeX, Markdown math, or mathematical notation.
                - NEVER add bold, italic, emojis or decorative formatting unless present in the document.
                - Keep abbreviations, symbols and units unchanged.
                - If text is unreadable, write [UNCLEAR] instead of guessing.

                Return only valid Markdown.
            """)

        # Build a lightweight OCR transcript
        for page in document.pages:

            if len(document.pages) > 1:
                prompt.append(f"\n--- PAGE {page.page_number} ---\n")

            for block in page.blocks:

                if not block.text.strip():
                    continue

                prompt.append(block.text.strip())

            prompt.append("")

        return "\n".join(prompt)    