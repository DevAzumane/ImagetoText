import json
from pathlib import Path
from dataclasses import asdict


class ChunkSerializer:

    @staticmethod
    def save(chunks, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = []

        for chunk in chunks:

            data.append(
                asdict(chunk)
            )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )