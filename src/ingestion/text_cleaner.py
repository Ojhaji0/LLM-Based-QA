"""
Text Cleaner — Remove layout artifacts common in institutional PDFs.
Handles: headers/footers, watermarks, ligatures, hyphenation, non-printable chars.
"""
from __future__ import annotations
import re
from collections import Counter


# Common ligature artifacts from PDF glyph encoding
LIGATURE_MAP = {
    "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb00": "ff", "\ufb05": "st",
}

# Non-printable / control characters (except tab, newline)
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")

# Watermark / confidentiality patterns
WATERMARK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"confidential", r"draft\s+only", r"do\s+not\s+distribute",
        r"for\s+internal\s+use", r"©\s*\d{4}", r"all\s+rights\s+reserved",
    ]
]


def clean_blocks(blocks: list[dict], remove_headers_footers: bool = True) -> list[dict]:
    """Clean a list of text blocks extracted from PDFs."""
    if remove_headers_footers:
        blocks = _remove_repeated_strings(blocks)

    cleaned = []
    for block in blocks:
        text = block["text"]
        text = _replace_ligatures(text)
        text = _remove_non_printable(text)
        text = _remove_page_numbers(text)
        text = _fix_hyphenation(text)
        text = _normalize_whitespace(text)
        
        # Drop if purely numeric or too short (likely page numbers or noise)
        if text.strip().isdigit() or len(text.strip()) < 15:
            continue
            
        block["text"] = text
        cleaned.append(block)

    return cleaned

def _remove_page_numbers(text: str) -> str:
    """Remove patterns like 'Page 1 of 10' or lone page numbers."""
    text = re.sub(r"(?i)page\s*\d+\s*(of\s*\d+)?", "", text)
    return text


def _replace_ligatures(text: str) -> str:
    for char, replacement in LIGATURE_MAP.items():
        text = text.replace(char, replacement)
    return text


def _remove_non_printable(text: str) -> str:
    return NON_PRINTABLE_RE.sub("", text)


def _fix_hyphenation(text: str) -> str:
    """Join words split across lines by hyphenation."""
    return re.sub(r"-\n(\w)", r"\1", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_strings(blocks: list[dict], threshold: int = 3) -> list[dict]:
    """
    Detect and remove strings that repeat across 3+ pages
    (header/footer/watermark pattern detection).
    """
    text_counts = Counter(b["text"].strip()[:100] for b in blocks)
    repeated = {t for t, c in text_counts.items() if c >= threshold}
    return [b for b in blocks if b["text"].strip()[:100] not in repeated]
