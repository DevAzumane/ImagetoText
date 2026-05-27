import ollama

from configs.settings import get_settings
from engine.prompts import (
    TRANSCRIPTION_PROMPT,
    build_summary_prompt
)

settings = get_settings()

class QwenVLEngine:

    def __init__(self, timeout=None):
        if timeout is None:
            self.client = ollama.Client()
        else:
            self.client = ollama.Client(timeout=timeout)

    def extract_text(self, image_path):
   
        response = self.client.chat(
            model=settings.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": TRANSCRIPTION_PROMPT,
                "images": [image_path]
            }]
        )

        return response["message"]["content"]

    def summarize_text(self, text, num_lines=3):

        prompt = build_summary_prompt(text, num_lines)

        response = self.client.chat(
            model=settings.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response["message"]["content"]
