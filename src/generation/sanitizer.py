"""
Context Sanitizer — guards against prompt injection in retrieved contexts.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"reveal (your |the |system )?prompt",
    r"you are now",
    r"disregard (the |all )?above",
    r"do not (follow|use)",
]

def sanitize_context(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text
