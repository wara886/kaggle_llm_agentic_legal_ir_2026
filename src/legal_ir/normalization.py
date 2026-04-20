import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
PUNCT_TRIM_RE = re.compile(r"[;,，。]+$")


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00a0", " ")
    text = text.replace("’", "'").replace("`", "'")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def normalize_citation(citation: str) -> str:
    text = normalize_text(citation)
    text = PUNCT_TRIM_RE.sub("", text)
    text = re.sub(r"\bArt\s+\.", "Art.", text)
    text = re.sub(r"\bAbs\s+\.", "Abs.", text)
    text = re.sub(r"\bE\s+\.", "E.", text)
    return text


def split_citations(serialized: str) -> list[str]:
    if not serialized:
        return []
    parts = [normalize_citation(x) for x in serialized.split(";")]
    return [x for x in parts if x]

