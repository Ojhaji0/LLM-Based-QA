"""
RAGAS Evaluation — framework for scoring answer quality.
"""
from loguru import logger
import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# We will use the pipeline to generate answers for the questions
from src.pipeline.rag_chain import RAGPipeline

def evaluate_system(qa_file: str = "data/eval/qa_pairs_synthetic.json"):
    # Note: RAGAS primarily expects OpenAI API key.
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY environment variable is missing. RAGAS requires OpenAI to act as the LLM judge for evaluation metrics.")
        logger.warning("Please set your API key (e.g., set OPENAI_API_KEY=sk-...) and run this script again.")
        return
        
    if not os.path.exists(qa_file):
        logger.error(f"QA dataset not found: {qa_file}. Run dataset_generator.py first.")
        return
        
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
        
    rag = RAGPipeline()
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    logger.info(f"Evaluating {len(qa_pairs)} questions through our RAG system...")
    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        logger.info(f"[{i+1}/{len(qa_pairs)}] Answering: {q}")
        
        # Use our retriever pipeline to get answers and contexts
        retrieved_chunks = rag.retriever.retrieve(q, top_k=5)
        context_texts = [chunk[0] for chunk in retrieved_chunks]
        
        # Ask our local LLM
        response = rag.ask(q)
        
        questions.append(q)
        answers.append(response["answer"])
        contexts.append(context_texts)
        ground_truths.append(pair["ground_truth"])
        
    # Format for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    logger.info("Running RAGAS metrics using OpenAI as the judge...")
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
    )
    
    df = result.to_pandas()
    out_path = "data/eval/ragas_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    
    logger.info(f"Evaluation complete. Results saved to {out_path}")
    logger.info(f"Aggregate Scores:\n{result}")

if __name__ == "__main__":
    evaluate_system()
