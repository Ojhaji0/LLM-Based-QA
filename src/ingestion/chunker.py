"""
Chunker — Recursive Character Text Splitting (primary) + Fixed-size (baseline B).
Tables are never split — kept as single chunks.
"""
from __future__ import annotations
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


def get_recursive_splitter(chunk_size: int = 600, chunk_overlap: int = 100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def get_fixed_splitter(chunk_size: int = 600):
    """Baseline B: Fixed-size with no overlap or separator awareness."""
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator="",
    )


def chunk_blocks(blocks: list[dict], strategy: str = "recursive",
                 chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict]:
    """
    Chunk a list of cleaned text blocks into indexable units.
    - Tables (is_table=True) are always kept as single chunks.
    - Text blocks are split according to chosen strategy.
    Returns list of chunk dicts with metadata.
    """
    splitter = (
        get_recursive_splitter(chunk_size, chunk_overlap)
        if strategy == "recursive"
        else get_fixed_splitter(chunk_size)
    )

    chunks = []
    for block in blocks:
        meta = {
            "source_file": os.path.basename(block.get("source_file", "")),
            "source_path": block.get("source_file", ""),
            "page": block.get("page", 0),
            "is_table": block.get("is_table", False),
            "extraction_method": block.get("extraction_method", "pymupdf"),
            "section": block.get("section", ""),
            "is_heading": block.get("is_heading", False),
        }

        if block.get("is_table"):
            # Tables: never split — single chunk
            chunks.append({**meta, "text": f"Document: {meta['source_file']}\n{block['text']}", "chunk_index": 0})
        else:
            docs = splitter.create_documents([block["text"]], metadatas=[meta])
            for i, doc in enumerate(docs):
                chunk_meta = {**doc.metadata, "chunk_index": i}
                chunks.append({**chunk_meta, "text": f"Document: {meta['source_file']}\n{doc.page_content}"})

    return chunks
