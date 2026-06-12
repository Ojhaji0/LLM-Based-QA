from src.pipeline.rag_chain import RAGPipeline
import sys

# Windows terminal fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

rag = RAGPipeline()

# Test Questions
questions = [
    "Who are the authors of this paper?",
    "What is the significance of the Transformer model?"
]

for q in questions:
    print(f"\nSawaal (Question): {q}")
    print("Thinking...")
    res = rag.ask(q)
    print(f"\nJawab (Answer):\n{res['answer']}")
    print("-" * 50)
    print(f"Latency: {res['latency']:.2f}s | Docs: {res['retrieved_docs']}")
