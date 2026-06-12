"""
Dataset Generator — generate synthetic QA pairs using an LLM.
"""
from loguru import logger
import json
import random
from src.retrieval.vector_store import get_vector_store
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def generate_qa_pairs(output_file: str = "data/eval/qa_pairs_synthetic.json", num_samples: int = 5):
    logger.info("Initializing Dataset Generator...")
    vectorstore = get_vector_store()
    
    # Get some random chunks from vector store
    data = vectorstore.get()
    docs = data['documents']
    
    if not docs:
        logger.error("No documents found in vectorstore. Run ingestion first.")
        return
        
    llm = Ollama(model="llama3.2", temperature=0.7)
    
    prompt = PromptTemplate.from_template(
        "You are an expert evaluator. Given the following context extracted from a document, "
        "generate one realistic question that a user might ask based on this context, "
        "and provide the factual answer.\n\n"
        "Context:\n{context}\n\n"
        "Output ONLY a valid JSON object with 'question' and 'answer' keys. Do not include any other text."
    )
    
    chain = prompt | llm
    
    qa_pairs = []
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Randomly select chunks
    selected_docs = random.sample(docs, min(num_samples, len(docs)))
    
    logger.info(f"Generating {len(selected_docs)} synthetic QA pairs using local model...")
    for i, doc in enumerate(selected_docs):
        try:
            res = chain.invoke({"context": doc})
            # Clean up JSON string if LLM returned markdown blocks
            res = res.strip().replace('```json', '').replace('```', '')
            parsed = json.loads(res)
            qa_pairs.append({
                "question": parsed["question"],
                "ground_truth": parsed["answer"],
                "source_context": doc
            })
            logger.info(f"Generated {i+1}/{len(selected_docs)}")
        except Exception as e:
            logger.warning(f"Failed to parse generation {i}: {e}")
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4)
        
    logger.info(f"Successfully saved {len(qa_pairs)} pairs to {output_file}")

if __name__ == "__main__":
    generate_qa_pairs()
