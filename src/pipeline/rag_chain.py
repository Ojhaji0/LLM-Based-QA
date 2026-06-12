"""
RAG Chain — orchestrates the end-to-end question answering pipeline.
"""
from loguru import logger
from src.retrieval.retriever import HybridRetrieverPipeline
from src.generation.prompt_builder import build_prompt
from src.generation.llm_backend import get_llm
from src.generation.citation_validator import validate_citations
import time
import os

# ---------------------------------------------------------------------------
# SOURCE ROUTING MAP
# Maps user-facing paper aliases → exact source_file value in Chroma metadata.
# Add new entries here whenever a new PDF is ingested.
# ---------------------------------------------------------------------------
SOURCE_ROUTING_MAP: dict[str, str] = {
    # Attention Is All You Need (Vaswani et al., 2017)
    "attention is all you need":      "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "transformer paper":              "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "nips 2017":                      "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "vaswani":                        "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "attention paper":                "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "nips-2017":                      "NIPS-2017-attention-is-all-you-need-Paper.pdf",
    # System Design Interview book
    "system design interview":        "SystemDesignInterview.pdf",
    "system design book":             "SystemDesignInterview.pdf",
    "systemdesigninterview":          "SystemDesignInterview.pdf",
}

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        self.retriever = HybridRetrieverPipeline()
        self.llm = get_llm(model_name="phi3:latest")
    def ask(self, query: str):
        t0 = time.time()
        clean_query = query.lower().strip()

        # --- LAYER 0: SOURCE-SPECIFIC ROUTING ---
        # If the user mentions a specific paper/book name, lock retrieval to that
        # source_file so we never mix chunks from different documents.
        source_filter: str | None = None
        for alias, filename in SOURCE_ROUTING_MAP.items():
            if alias in clean_query:
                source_filter = filename
                logger.info(f"Source routing: alias '{alias}' → '{filename}'")
                break

        # --- LAYER 1: QUERY EXPANSION ---
        EXPANSIONS = {
            "rag": "Retrieval-Augmented Generation (RAG) framework architecture",
            # Added P100/NVIDIA/training schedule so vector search hits the NIPS hardware section
            "hardware": "hardware setup CPU GPU RAM P100 NVIDIA training schedule floating-point capacity",
            "abstract": "abstract summary overview of the paper",
            "limitations": "limitations constraints weaknesses future work",
            "methodology": "methodology approach experimental design",
            "results": "results performance metrics evaluation BLEU EN-DE EN-FR",
            "author": "authors names contributors researchers who wrote",
            "who wrote": "authors names contributors researchers",
            "bleu": "BLEU score machine translation results performance WMT",
            "dataset": "training data WMT English German French sentence pairs",
            "outperform": "Transformer vs RNN recurrent advantages parallelization",
            "training": "training schedule steps warmup Adam optimizer learning rate",
        }
        for kw, exp in EXPANSIONS.items():
            if kw in clean_query:
                logger.info(f"Expanding query for keyword '{kw}': {exp}")
                query = f"{query} ({exp})"
                break

        # --- SOURCE-AWARE SECONDARY EXPANSION ---
        # When locked to a specific source, append a document-specific context hint.
        # This helps BM25 and dense retrieval use the right vocabulary for that doc.
        SOURCE_VOCAB_HINTS: dict[str, str] = {
            "NIPS-2017-attention-is-all-you-need-Paper.pdf": (
                "Transformer self-attention multi-head scaled dot-product "
                "encoder decoder P100 GPU WMT BLEU Vaswani 2017"
            ),
            "SystemDesignInterview.pdf": (
                "system design scalability load balancer consistent hashing "
                "CAP theorem database sharding rate limiter cache"
            ),
        }
        if source_filter and source_filter in SOURCE_VOCAB_HINTS:
            hint = SOURCE_VOCAB_HINTS[source_filter]
            query = f"{query} [{hint}]"
            logger.info(f"Source vocab hint applied for '{source_filter}'")

        # --- LAYER 2: SECTION-AWARE ROUTING ---
        # Combine with source filter when both apply.
        metadata_filter: dict | None = None
        SECTION_MAP = {
            "abstract": "ABSTRACT",
            "introduction": "INTRODUCTION",
            "methodology": "METHODOLOGY",
            "experimental": "METHODOLOGY",
            "results": "RESULTS",
            "conclusion": "CONCLUSION",
            "future work": "FUTURE WORK",
            "limitations": "LIMITATIONS",
        }
        for kw, section_tag in SECTION_MAP.items():
            if kw in clean_query:
                logger.info(f"Section keyword '{kw}' detected. Filter section='{section_tag}'")
                metadata_filter = {"section": section_tag}
                break

        # Merge source filter into metadata_filter
        # Chroma supports only single-key $eq filters natively; we apply source
        # as a post-retrieval guard when a section filter is already active.
        if source_filter and metadata_filter:
            # Both active: run with section filter, then post-filter by source
            logger.info(f"Both source & section filters active — will post-filter by source.")
        elif source_filter:
            metadata_filter = {"source_file": source_filter}

        # --- LAYER 3: DYNAMIC TOP_K ---
        # For summarization tasks, we need more context chunks to capture whole sections
        is_summary = "summarize" in clean_query or "summary" in clean_query
        top_k_chunks = 15 if is_summary else 5

        # 1. Retrieve
        logger.info(f"Processing query: '{query}' | filter={metadata_filter} | source={source_filter}")
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k_chunks, metadata_filter=metadata_filter)

        # Post-filter by source when both section + source filters are active
        # (Chroma can't AND two metadata fields directly in one filter call)
        if source_filter and retrieved_chunks:
            filtered = [c for c in retrieved_chunks if c[1].get("source_file") == source_filter]
            if filtered:
                logger.info(f"Post-source-filter: {len(retrieved_chunks)} → {len(filtered)} chunks from '{source_filter}'")
                retrieved_chunks = filtered
            else:
                logger.warning(f"Post-source-filter returned 0 results for '{source_filter}'. Keeping unfiltered results.")

        # Fallback: if filter returned zero chunks, retry without filter
        if not retrieved_chunks and metadata_filter:
            logger.warning(f"Filter {metadata_filter} returned zero results. Falling back to un-filtered search.")
            retrieved_chunks = self.retriever.retrieve(query, top_k=top_k_chunks)
        
        # 2. Build Prompt
        prompt = build_prompt(query, retrieved_chunks)
        top_score = retrieved_chunks[0][2] if retrieved_chunks else 0.0
        
        # 3. Generate
        logger.info("Generating answer with LLM...")
        answer = self.llm.invoke(prompt)
        
        # 4. Validate Citations
        source_files = list(set([os.path.basename(chunk[1].get('source_file', '')) for chunk in retrieved_chunks]))
        validation = validate_citations(answer, source_files)
        
        latency = time.time() - t0
        logger.info(f"Answered in {latency:.2f}s")
        
        return {
            "query": query,
            "answer": answer,
            "citations_valid": validation["valid"],
            "citations_invalid": validation["invalid"],
            "hallucinated_citations": validation["hallucinated_citations"],
            "latency": latency,
            "retrieved_docs": len(retrieved_chunks),
            "top_score": float(top_score)
        }

    def _reject_response(self, query, message, start_time):
        return {
            "query": query,
            "answer": message,
            "citations_valid": [],
            "citations_invalid": [],
            "hallucinated_citations": 0,
            "latency": time.time() - start_time,
            "retrieved_docs": 0,
            "top_score": 0.0
        }

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    rag = RAGPipeline()
    response = rag.ask("What is attention?")
    print("\n\n=== FINAL ANSWER ===")
    print(response["answer"])
    print("\n--- Validation ---")
    print(f"Valid Citations: {response['citations_valid']}")
    print(f"Invalid Citations: {response['citations_invalid']}")
    print(f"Latency: {response['latency']:.2f}s")
