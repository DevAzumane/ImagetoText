import base64
from pathlib import Path
from utils.image_utils import ImageUtils
import os

import requests

from config import (
    LLAMA_SERVER_URL,
    QWEN_MODEL_NAME,
    QWEN_MAX_TOKENS,
    QWEN_TEMPERATURE,
    QWEN_IMAGE_SIZE
)


class QwenModel:

    def __init__(self):

        self.endpoint = (
            f"{LLAMA_SERVER_URL}/v1/chat/completions"
        )

    # --------------------------------------------------------
    # Encode image to Base64
    # --------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:

        image_path = Path(image_path)

        with open(image_path, "rb") as image:

            encoded = base64.b64encode(
                image.read()
            ).decode("utf-8")

        return encoded

    # --------------------------------------------------------
    # Generate Response
    # --------------------------------------------------------

    def generate(

        self,

        prompt: str,

        image_paths: list[str],

    ) -> dict:

        content = [

            {

                "type": "text",

                "text": prompt,

            }

        ]

        for image_path in image_paths:

            resized_image = ImageUtils.resize_for_qwen(
                image_path,
                max_size=QWEN_IMAGE_SIZE
            )

            try:

                encoded = self._encode_image(
                    resized_image
                )

            finally:

                if os.path.exists(resized_image):
                    os.remove(resized_image)

            content.append(

                {

                    "type": "image_url",

                    "image_url": {

                        "url": (
                            "data:image/jpeg;base64,"
                            + encoded
                        )

                    }

                }

            )

        payload = {

            "model": QWEN_MODEL_NAME,

            "messages": [

                {

                    "role": "user",

                    "content": content,

                }

            ],

            "temperature": QWEN_TEMPERATURE,

            "max_tokens": QWEN_MAX_TOKENS,

        }

        response = requests.post(

            self.endpoint,

            json=payload,

            timeout=300,

        )

        response.raise_for_status()

        return response.json()