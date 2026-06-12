"""
PDF Loader — Primary: PyMuPDF (fitz)
Extracts text blocks with page number and source metadata.
Falls back to OCR for pages with insufficient text.
"""
from __future__ import annotations
import fitz  # PyMuPDF
from loguru import logger


MIN_TEXT_CHARS = 50  # pages below this threshold trigger OCR fallback


def load_pdf(filepath: str, use_ocr_fallback: bool = True) -> list[dict]:
    """
    Load a PDF and return a list of text blocks.
    Each block: {text, page, source_file, extraction_method}
    """
    doc = fitz.open(filepath)
    blocks = []
    ocr_pages = 0

    for page_num, page in enumerate(doc):
        raw_text = page.get_text().strip()

        # OCR fallback for scanned/image pages
        if use_ocr_fallback and len(raw_text) < MIN_TEXT_CHARS:
            text = _ocr_page(page)
            method = "ocr"
            ocr_pages += 1
        else:
            method = "pymupdf"
            current_section = "" # Keep track of the current active section
            
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                
                block_text = []
                is_heading = False
                for line in block["lines"]:
                    for span in line["spans"]:
                        txt = span["text"].strip()
                        if not txt: continue
                        # Heuristic: Larger font or bold usually indicates a heading
                        if span["size"] > 12 or (span["flags"] & 2**4): # bit 4 is bold in fitz
                            if len(txt) < 100: # headings are usually short
                                is_heading = True
                        block_text.append(txt)
                
                text = " ".join(block_text).strip()
                if is_heading:
                    current_section = text
                    
                if text:
                    blocks.append({
                        "text": text,
                        "page": page_num + 1,
                        "source_file": filepath,
                        "extraction_method": method,
                        "is_table": False,
                        "is_heading": is_heading,
                        "section": current_section # Propagate section to content
                    })
            continue  # already appended per-block, skip below

        if text.strip():
            blocks.append({
                "text": text,
                "page": page_num + 1,
                "source_file": filepath,
                "extraction_method": method,
                "is_table": False,
            })

    doc.close()
    if ocr_pages:
        logger.warning(f"{filepath}: {ocr_pages} pages required OCR fallback")
    return blocks


def _ocr_page(page) -> str:
    """Render page to image and run Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""
