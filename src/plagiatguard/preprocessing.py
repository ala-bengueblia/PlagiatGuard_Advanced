import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_passages(
    text: str,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[str]:

    text = re.sub(r"\s+", " ", text or "").strip()

    if not text:
        return []

    passages = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            boundary = text.rfind(". ", start, end)

            if boundary > start + max_chars // 2:
                end = boundary + 1

        passages.append(text[start:end].strip())

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return passages