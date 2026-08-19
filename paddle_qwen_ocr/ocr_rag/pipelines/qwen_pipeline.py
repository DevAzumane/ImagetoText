from services.qwen_service import QwenService


class QwenPipeline:

    def __init__(self):

        self.service = QwenService()

    def run(
        self,
        image_path,
        document,
    ):

        markdown = self.service.analyze(
            image_path=image_path,
            document=document,
        )

        document.enhanced_markdown = markdown

        # -------------------------
        # Extract document title
        # -------------------------

        for line in markdown.splitlines():

            line = line.strip()

            if line.startswith("# "):

                document.metadata["title"] = line.replace("# ", "")

                break

        return document