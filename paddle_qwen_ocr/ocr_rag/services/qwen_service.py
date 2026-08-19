from models.qwen.qwen_model import QwenModel
from models.qwen.prompt_builder import PromptBuilder


class QwenService:

    def __init__(self):

        self.model = QwenModel()

    def analyze(

        self,

        image_path,

        document,

    ):

        prompt = PromptBuilder.build(
            document
        )

        response = self.model.generate(

            prompt=prompt,

            image_paths=[image_path],

        )

        return response["choices"][0]["message"]["content"]