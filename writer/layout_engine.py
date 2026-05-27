from pathlib import Path
import re


class LayoutEngine:

    """
    AI-driven layout engine.

    Converts raw OCR/LLM summaries into structured layout blocks
    that the handwriting renderer can directly render.

    ------------------------------------------------------------

    Example input file:

    TITLE : Paragraph

    img_1: This handout explains paragraphs.
    img_2: Decide on an argument and thesis.

    ------------------------------------------------------------

    Output blocks:

    [
        {"type": "title", "text": "Paragraph"},

        {"type": "heading", "text": "img_1"},
        {"type": "paragraph", "text": "This handout explains paragraphs."},

        {"type": "blank", "text": ""},

        {"type": "heading", "text": "img_2"},
        {"type": "paragraph", "text": "Decide on an argument and thesis."},
    ]
    """

    # =========================================================
    # PUBLIC
    # =========================================================

    @classmethod
    def build(
        cls,
        input_dir="data/output",
        template="default",
        custom_title=None,
    ):

        layout_text = cls._generate_layout_text(
            input_dir=input_dir,
            template=template,
        )

        print("\n========== RAW LAYOUT ==========\n")
        print(layout_text)
        print("\n================================\n")

        blocks = cls.parse_layout_text(layout_text)

        # -----------------------------------------
        # Optional user custom title override
        # -----------------------------------------

        if custom_title:

            for block in blocks:

                if block["type"] == "title":
                    block["text"] = custom_title
                    break

        print("\n========== BLOCKS ==========\n")

        for block in blocks:
            print(block)

        print("\n============================\n")

        return blocks

    # =========================================================
    # GENERATE RAW LAYOUT TEXT
    # =========================================================
    @classmethod
    def _generate_layout_text(
        cls,
        input_dir="data/output",
        template="default",
    ):

        input_path = Path(input_dir)

        # -----------------------------
        # FILE SUPPORT
        # -----------------------------

        if input_path.is_file():

            return input_path.read_text(
                encoding="utf-8"
            )

        # -----------------------------
        # DIRECTORY SUPPORT
        # -----------------------------

        txt_files = sorted(input_path.glob("*.txt"))

        if not txt_files:
            return "TITLE: Empty Notes"

        summaries = []
        saw_pipeline_output = False

        for file in txt_files:
            content = file.read_text(
                encoding="utf-8"
            ).strip()

            if "===== SUMMARY =====" in content:
                saw_pipeline_output = True
                summaries.append(
                    f"{file.stem}: {cls._extract_summary(file)}"
                )

        if saw_pipeline_output:
            return "TITLE: Notes\n" + "\n".join(summaries)

        full_text = []

        for file in txt_files:

            content = file.read_text(
                encoding="utf-8"
            ).strip()

            full_text.append(content)

        return "\n\n".join(full_text)

    # =========================================================
    # PARSE RAW LAYOUT TEXT
    # =========================================================

    @classmethod
    def parse_layout_text(cls, text):

        blocks = []

        lines = text.splitlines()

        for line in lines:

            line = line.strip()

            # -------------------------------------------------
            # EMPTY LINE
            # -------------------------------------------------

            if not line:
                continue

            # -------------------------------------------------
            # TITLE
            # -------------------------------------------------

            if line.upper().startswith("TITLE"):

                parts = line.split(":", 1)

                if len(parts) < 2:
                    continue

                title = parts[1].strip()

                blocks.append({
                    "type": "title",
                    "text": title
                })

                continue

            # -------------------------------------------------
            # IMAGE + SUMMARY
            # -------------------------------------------------

            if ":" in line:

                heading, content = line.split(":", 1)

                heading = heading.strip()

                content = content.strip()

                # IMAGE NAME
                blocks.append({
                    "type": "heading",
                    "text": heading
                })

                # SUMMARY
                blocks.append({
                    "type": "paragraph",
                    "text": content
                })

                # SPACE AFTER EACH IMAGE BLOCK
                blocks.append({
                    "type": "blank",
                    "text": ""
                })

                continue

        return blocks

    # =========================================================
    # EXTRACT SUMMARY
    # =========================================================

    @classmethod
    def _extract_summary(cls, path):

        content = path.read_text(
            encoding="utf-8"
        ).strip()

        # -----------------------------------------------------
        # REMOVE TITLE LINE
        # -----------------------------------------------------

        content = re.sub(
            r"TITLE\s*:\s*.+",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = content.strip()

        # -----------------------------------------------------
        # SUMMARY MARKER
        # -----------------------------------------------------

        marker = "===== SUMMARY ====="

        if marker in content:

            summary = content.split(
                marker,
                1
            )[1].strip()

        else:
            summary = content

        # -----------------------------------------------------
        # CLEANUP
        # -----------------------------------------------------

        summary = " ".join(summary.split())

        summary = (
            summary
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("–", "-")
            .replace("—", "-")
        )

        return summary

    # =========================================================
    # DEBUG PREVIEW
    # =========================================================

    @classmethod
    def preview(
        cls,
        input_dir="data/output",
        template="default",
        custom_title=None,
    ):

        blocks = cls.build(
            input_dir=input_dir,
            template=template,
            custom_title=custom_title,
        )

        print("\n========== FINAL BLOCKS ==========\n")

        for block in blocks:
            print(block)

        print("\n==================================\n")
