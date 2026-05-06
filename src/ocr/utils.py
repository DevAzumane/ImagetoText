from src.ocr.config import IMAGE_HEIGHT, IMAGE_WIDTH

# Character set (keep it simple for now)
chars = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?-:/()'\" "
)

char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is reserved for CTC blank
idx_to_char = {i + 1: c for i, c in enumerate(chars)}


def encode(text):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode(indices):
    result = []
    prev = -1

    for i in indices:
        if i != prev and i != 0:
            result.append(idx_to_char.get(i, ""))
        prev = i

    return "".join(result)
