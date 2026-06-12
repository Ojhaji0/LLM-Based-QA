"""
Prompt Builder — dynamic assembly of context, query, and system instructions.
"""
from __future__ import annotations
from src.generation.sanitizer import sanitize_context

def build_prompt(query: str, retrieved_chunks: list[tuple]) -> str:
    """
    Constructs the prompt with context.
    retrieved_chunks: list of (doc_text, metadata, score, id)
    """
    system_instruction = (
        "You are a helpful and intelligent document-grounded QA assistant.\n\n"
        "GUIDELINES:\n"
        "1. Base your answer on the retrieved context provided below. Be as helpful as possible.\n"
        "2. KEEP ANSWERS CONCISE and to the point, unless the user explicitly asks for a detailed explanation.\n"
        "3. If the context contains partial information related to the question, provide what is available.\n"
        "4. If the context is completely irrelevant, state that you cannot answer based on the provided documents.\n"
        "5. DO NOT make up information or facts that are not present in the context.\n"
        "6. CITATIONS: Do NOT place citations inline. Instead, list all sources used in a single block at the very end of your answer in the format:\n"
        "   [Sources: <filename1> (Page <X>), <filename2> (Page <Y>)]\n"
        "7. ALWAYS maintain a professional, objective tone.\n\n"
    )
    
    context_str = "--- RETRIEVED CONTEXT ---\n"
    for i, (doc_text, metadata, _, _) in enumerate(retrieved_chunks):
        source = metadata.get("source_file", "Unknown")
        # Ensure we only use the base name for citation brevity
        if "/" in source or "\\" in source:
            import os
            source = os.path.basename(source)
            
        page = metadata.get("page", "?")
        sanitized_text = sanitize_context(doc_text)
        context_str += f"Document [{i+1}] | Source: {source} | Page: {page}\n{sanitized_text}\n\n"
        
    context_str += "--- END CONTEXT ---\n\n"
    
    user_prompt = f"Question: {query}\nAnswer:"
    
    return system_instruction + context_str + user_prompt
