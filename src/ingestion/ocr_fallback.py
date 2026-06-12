"""
OCR Fallback — pytesseract for scanned pages
"""
import pytesseract
from PIL import Image
import fitz

def has_extractable_text(page, min_chars=50):
    return len(page.get_text().strip()) >= min_chars

def ocr_page(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)

def load_pdf_with_ocr_fallback(filepath):
    doc = fitz.open(filepath)
    blocks = []
    for page_num, page in enumerate(doc):
        if has_extractable_text(page):
            text = page.get_text()
        else:
            text = ocr_page(page)   # scanned page fallback
        blocks.append({"text": text, "page": page_num+1,
                        "source_file": filepath})
    return blocks
