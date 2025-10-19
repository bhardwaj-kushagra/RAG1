import regex as re
from typing import List


_sent_end_re = re.compile(r"([.!?]+\s+|\n+)" , flags=re.MULTILINE)
_multi_space_re = re.compile(r"\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence enders and newlines, keep chunks
    parts = _sent_end_re.split(text)
    sents = []
    buf = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        end = parts[i+1] if i + 1 < len(parts) else ""
        buf = (chunk + (end.strip()[:1] if end.strip() else "")).strip()
        if buf:
            sents.append(_multi_space_re.sub(" ", buf))
    # Fallback: if no delimiter split happened, return the whole text
    if not sents:
        sents = [_multi_space_re.sub(" ", text)]
    return sents


_word_re = re.compile(r"\p{L}[\p{L}\p{Mn}\p{Nd}\-']*|\d+(?:[.,]\d+)*")


def tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in _word_re.finditer(text)]
