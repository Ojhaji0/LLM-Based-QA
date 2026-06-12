"""
PDF Loader — Secondary: pdfplumber
Table-aware extraction: tables serialized as Markdown, text extracted separately.
Use for documents known to contain complex tables (GPA scales, fee matrices, etc.)
"""
from __future__ import annotations
import pdfplumber
from loguru import logger


def load_pdf_pdfplumber(filepath: str) -> list[dict]:
    """
    Load a PDF using pdfplumber.
    Tables are preserved as Markdown — never split mid-row.
    Returns list of blocks: {text, page, source_file, is_table}
    """
    blocks = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # --- Extract tables first ---
            table_bboxes = []
            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    blocks.append({
                        "text": md,
                        "page": page_num + 1,
                        "source_file": filepath,
                        "extraction_method": "pdfplumber_table",
                        "is_table": True,
                    })
                    table_bboxes.append(True)

            # --- Extract remaining text (non-table regions) ---
            text = page.extract_text()
            if text and text.strip():
                blocks.append({
                    "text": text.strip(),
                    "page": page_num + 1,
                    "source_file": filepath,
                    "extraction_method": "pdfplumber_text",
                    "is_table": False,
                })

    logger.info(f"pdfplumber: {filepath} — {len(blocks)} blocks extracted")
    return blocks


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a Markdown table string."""
    if not table or not table[0]:
        return ""
    rows = [[str(cell).strip() if cell else "" for cell in row] for row in table]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join([header, separator, body])
