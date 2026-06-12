"""
model_benchmark.py — Multi-Model RAG Benchmark Runner
======================================================
Tests multiple Ollama models on the same question set using the full RAG
pipeline (Hybrid Retrieval + Reranking + LLM Generation).

Usage:
    python model_benchmark.py
    python model_benchmark.py --questions "What is attention?" "Explain BLEU score"
    python model_benchmark.py --models llama3.2 phi3

Results saved to: logs/benchmark_results.json
"""

import sys
import json
import time
import argparse
import os
from datetime import datetime
from loguru import logger

# Fix Windows Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "phi3:latest",       # Default — slightly faster, detailed answers
    "llama3.2:latest",   # Fallback — smaller, good for quick queries
]

DEFAULT_QUESTIONS = [
    "What is attention mechanism and how does it work?",
    "Explain the results of the Transformer model compared to RNNs.",
    "What hardware was used in the experiments?",
    "What are the limitations of the proposed approach?",
    "Summarize the abstract of the paper.",
]

# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(models: list[str], questions: list[str]) -> list[dict]:
    from src.retrieval.retriever import HybridRetrieverPipeline
    from src.generation.llm_backend import get_llm
    from src.generation.prompt_builder import build_prompt
    from src.generation.citation_validator import validate_citations

    logger.info("Initializing shared retriever pipeline (used by all models)...")
    retriever = HybridRetrieverPipeline()
    logger.info("Retriever ready.")

    all_results = []

    for model_tag in models:
        print(f"\n{'='*65}")
        print(f"  🤖  MODEL: {model_tag}")
        print(f"{'='*65}")

        try:
            llm = get_llm(backend="ollama", model_name=model_tag)
        except Exception as e:
            logger.error(f"Could not load model '{model_tag}': {e}")
            for q in questions:
                all_results.append({
                    "model": model_tag,
                    "question": q,
                    "answer": "ERROR: Model load failed",
                    "latency_s": None,
                    "answer_length": 0,
                    "docs_retrieved": 0,
                    "top_score": 0.0,
                    "error": str(e),
                })
            continue

        for q_idx, question in enumerate(questions, 1):
            print(f"\n  Q{q_idx}: {question}")
            print(f"  {'-'*60}")

            try:
                # Retrieve (shared pipeline)
                t_start = time.time()
                chunks = retriever.retrieve(question, top_k=5)
                prompt = build_prompt(question, chunks)
                top_score = float(chunks[0][2]) if chunks else 0.0

                # Generate
                answer = llm.invoke(prompt)
                latency = time.time() - t_start

                # Validate citations
                source_files = list({
                    os.path.basename(c[1].get("source_file", ""))
                    for c in chunks
                })
                validation = validate_citations(answer, source_files)

                # Display
                print(f"  ⏱  Latency  : {latency:.2f}s")
                print(f"  📄 Docs     : {len(chunks)} retrieved | Top Score: {top_score:.4f}")
                print(f"  ✅ Citations: valid={len(validation['valid'])} | hallucinated={validation['hallucinated_citations']}")
                print(f"  📝 Answer   : {answer[:250].strip()}{'...' if len(answer) > 250 else ''}")

                all_results.append({
                    "model": model_tag,
                    "question": question,
                    "answer": answer,
                    "latency_s": round(latency, 3),
                    "answer_length": len(answer),
                    "docs_retrieved": len(chunks),
                    "top_score": round(top_score, 4),
                    "citations_valid": validation["valid"],
                    "hallucinated_citations": validation["hallucinated_citations"],
                    "error": None,
                })

            except Exception as e:
                latency = time.time() - t_start if "t_start" in dir() else None
                logger.error(f"  ❌ Error on Q{q_idx} with {model_tag}: {e}")
                all_results.append({
                    "model": model_tag,
                    "question": question,
                    "answer": f"ERROR: {e}",
                    "latency_s": round(latency, 3) if latency else None,
                    "answer_length": 0,
                    "docs_retrieved": 0,
                    "top_score": 0.0,
                    "error": str(e),
                })

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict], models: list[str], questions: list[str]):
    """Prints a per-model aggregate comparison table."""

    print(f"\n\n{'='*75}")
    print("  📊  BENCHMARK SUMMARY — Aggregate per Model")
    print(f"{'='*75}")

    col_model    = 22
    col_latency  = 12
    col_ans_len  = 12
    col_docs     = 10
    col_score    = 11
    col_halluc   = 12

    header = (
        f"{'Model':<{col_model}} "
        f"{'Avg Latency':>{col_latency}} "
        f"{'Avg AnsLen':>{col_ans_len}} "
        f"{'Avg Docs':>{col_docs}} "
        f"{'Avg Score':>{col_score}} "
        f"{'Hallucinations':>{col_halluc}}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for model_tag in models:
        model_rows = [r for r in results if r["model"] == model_tag and r["error"] is None]
        if not model_rows:
            print(f"  {model_tag:<{col_model-2}} | NO VALID RESULTS (all errors)")
            continue

        avg_latency  = sum(r["latency_s"]      for r in model_rows) / len(model_rows)
        avg_ans_len  = sum(r["answer_length"]  for r in model_rows) / len(model_rows)
        avg_docs     = sum(r["docs_retrieved"] for r in model_rows) / len(model_rows)
        avg_score    = sum(r["top_score"]      for r in model_rows) / len(model_rows)
        total_halluc = sum(r["hallucinated_citations"] for r in model_rows)
        n_success    = len(model_rows)

        print(
            f"{model_tag:<{col_model}} "
            f"{avg_latency:>{col_latency}.2f}s "
            f"{avg_ans_len:>{col_ans_len}.0f} chars "
            f"{avg_docs:>{col_docs}.1f} "
            f"{avg_score:>{col_score}.4f} "
            f"{total_halluc:>{col_halluc}} "
            f"  [{n_success}/{len(questions)} OK]"
        )

    print(separator)

    # Per-question breakdown
    print(f"\n\n{'='*75}")
    print("  📋  Per-Question Latency Breakdown")
    print(f"{'='*75}")

    for q_idx, question in enumerate(questions, 1):
        short_q = (question[:55] + "...") if len(question) > 55 else question
        print(f"\n  Q{q_idx}: {short_q}")
        print(f"  {'Model':<22} {'Latency':>10} {'AnsLen':>10} {'TopScore':>10}")
        print(f"  {'-'*55}")
        for model_tag in models:
            row = next(
                (r for r in results if r["model"] == model_tag and r["question"] == question),
                None
            )
            if row is None or row["error"]:
                print(f"  {model_tag:<22} {'ERROR':>10}")
            else:
                print(
                    f"  {model_tag:<22} "
                    f"{row['latency_s']:>10.2f}s "
                    f"{row['answer_length']:>8} ch "
                    f"{row['top_score']:>10.4f}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: list[dict]):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"logs/benchmark_{timestamp}.json"

    payload = {
        "run_timestamp": timestamp,
        "total_runs": len(results),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Also write/overwrite the "latest" file for easy access
    latest_path = "logs/benchmark_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved → {out_path}")
    logger.info(f"Latest copy   → {latest_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple Ollama models on the RAG Q&A pipeline."
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="List of Ollama model tags to benchmark (e.g. llama3.2:latest phi3:latest)",
    )
    parser.add_argument(
        "--questions", nargs="+", default=DEFAULT_QUESTIONS,
        help="List of questions to test (wrap each in quotes)",
    )
    args = parser.parse_args()

    models    = args.models
    questions = args.questions

    print(f"\n{'='*65}")
    print(f"  🔬  LLM RAG Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"  Models    : {', '.join(models)}")
    print(f"  Questions : {len(questions)}")
    print(f"  Total Runs: {len(models) * len(questions)}")
    print(f"{'='*65}")

    results = run_benchmark(models=models, questions=questions)
    print_summary_table(results=results, models=models, questions=questions)
    saved_path = save_results(results)

    print(f"\n✅ Benchmark complete! Full results → {saved_path}\n")


if __name__ == "__main__":
    main()
