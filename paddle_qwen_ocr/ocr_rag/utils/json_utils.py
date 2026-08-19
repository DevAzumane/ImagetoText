import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any


class JSONSerializer:
    """
    Generic serializer for dataclass objects.
    """

    @staticmethod
    def to_dict(obj: Any) -> dict:

        if is_dataclass(obj):

            return asdict(obj)

        raise TypeError(
            f"{type(obj)} is not a dataclass."
        )

    @staticmethod
    def save(
        obj: Any,
        output_path: str | Path,
        indent: int = 4,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                JSONSerializer.to_dict(obj),
                f,
                indent=indent,
                ensure_ascii=False,
            )