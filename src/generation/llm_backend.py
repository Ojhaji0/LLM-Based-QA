"""
LLM Backend — Ollama integration. Supports swappable model selection.
"""
from langchain_community.llms import Ollama

def get_llm(backend: str = "ollama", model_name: str = "llama3.2"):
    """
    Returns an LLM instance.
    
    Args:
        backend:    "ollama" (default) — local Ollama server
        model_name: Any Ollama-compatible model tag, e.g. "llama3.2:3b",
                    "mistral:7b", "phi3:mini", "gemma2:2b"
    """
    if backend == "ollama":
        return Ollama(
            model=model_name,
            temperature=0.1,
            num_predict=512
        )
