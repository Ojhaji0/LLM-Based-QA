"""
Citation Validator — post-hoc check for hallucinated citations.
"""
from __future__ import annotations
import re

def validate_citations(answer: str, source_files: list[str]) -> dict:
    cited = re.findall(r"\[Source:\s*(.+?),\s*Page\s*(\d+)\]", answer)
    valid, invalid = [], []
    for filename, page in cited:
        if any(filename.strip() in sf for sf in source_files):
            valid.append((filename, page))
        else:
            invalid.append((filename, page))
    return {"valid": valid, "invalid": invalid,
            "hallucinated_citations": len(invalid)}
