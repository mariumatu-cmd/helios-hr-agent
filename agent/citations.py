"""Shared citation identity rules for answer validation and evaluation."""
import re
import unicodedata

_CITATION = re.compile(r"\b(POL-[A-Z]+-\d+)\s*§\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2212"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_DASHES)


def identities(text: str) -> set[str]:
    return {f"{doc.upper()} §{section}" for doc, section in _CITATION.findall(normalize(text))}


def cited_labels(answer: str, labels: list[str]) -> list[str]:
    used = identities(answer)
    return [label for label in labels if identities(label) & used]
