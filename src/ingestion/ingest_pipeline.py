"""
Ingestion Pipeline — Orchestrates loading, cleaning, chunking, and embedding of PDFs into Chroma DB.
"""
import os
import glob
from loguru import logger
from langchain_community.vectorstores import Chroma

# Import components
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.pdf_loader_pdfplumber import load_pdf_pdfplumber
from src.ingestion.docx_loader import load_docx
from src.ingestion.text_cleaner import clean_blocks
from src.ingestion.chunker import chunk_blocks
from src.ingestion.embedder import embed_texts

class IngestionPipeline:
    def __init__(self, raw_pdfs_dir: str = None, vectorstore_dir: str = None):
        self.raw_pdfs_dir = raw_pdfs_dir or os.path.join(os.getcwd(), "data", "raw_pdfs")
        self.vectorstore_dir = vectorstore_dir or os.path.join(os.getcwd(), "vectorstore")
        os.makedirs(self.raw_pdfs_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)

    def process_all(self):
        doc_files = []
        for ext in ["*.pdf", "*.docx", "*.doc"]:
            doc_files.extend(glob.glob(os.path.join(self.raw_pdfs_dir, ext)))
            
        if not doc_files:
            logger.warning(f"No documents found in {self.raw_pdfs_dir}")
            return

        all_chunks = []
        for filepath in doc_files:
            logger.info(f"Processing {filepath}...")
            
            blocks = []
            if filepath.lower().endswith(('.docx', '.doc')):
                blocks = load_docx(filepath)
            else:
                # Try PyMuPDF first for PDF
                try:
                    blocks = load_pdf(filepath)
                except Exception as e:
                    logger.error(f"PyMuPDF failed on {filepath}: {e}, trying pdfplumber...")
                    blocks = load_pdf_pdfplumber(filepath)
            
            # Clean text
            cleaned_blocks = clean_blocks(blocks)
            
            # Chunk
            chunks = chunk_blocks(cleaned_blocks, strategy="recursive")
            all_chunks.extend(chunks)

        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        if not all_chunks:
            return

        # Prepare for Chroma
        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = []
        ids = []
        for i, chunk in enumerate(all_chunks):
            # Create a deterministic ID based on content hash and source
            chunk_id = f"chunk_{hash(chunk['text'] + chunk.get('source_file', ''))}_{i}"
            ids.append(chunk_id)
            
            # remove 'text' from metadata and add the ID for retrieval syncing
            meta = {k: v for k, v in chunk.items() if k != "text"}
            meta["id"] = chunk_id
            metadatas.append(meta)

        # Generate Embeddings
        # Since we use SBERT, we can use a custom Langchain Embeddings wrapper or use Chroma directly
        from langchain_community.embeddings import HuggingFaceEmbeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},  # bge-small is fast enough on CPU, saving GPU for LLM
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

        logger.info(f"Upserting {len(texts)} chunks to Chroma DB at {self.vectorstore_dir}...")
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=hf_embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.vectorstore_dir
        )
        vectorstore.persist()
        logger.info("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.process_all()
