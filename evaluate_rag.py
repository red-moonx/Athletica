import json
import random
import os
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
ES_URL = "http://localhost:9200"
INDEX_NAME = "athletica_chunks"
CHUNKS_PATH = "data/processed/optimizations/optimization_3/chunks.json"
GROUND_TRUTH_PATH = "data/processed/optimizations/optimization_3/ground_truth_round_3.json"
OUTPUT_PATH = "data/processed/optimizations/optimization_3/rag_evaluation.json"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
LLM_MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 50

# Pydantic schema for LLM-as-a-judge evaluation
class EvaluationResult(BaseModel):
    faithfulness: int = Field(description="1 if the answer is entirely supported by the context, 0 if it contains unsupported claims or hallucinations")
    relevance: int = Field(description="1 if the answer directly addresses the user's question, 0 otherwise")
    reasoning: str = Field(description="Brief explanation of why these scores were given")

def search_knn(es, query: str, model, top_k: int = 5):
    vector = model.encode(query).tolist()
    search_body = {
        "knn": {
            "field": "dense_vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": 100
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body, size=top_k)
    return [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]

def generate_answer(client, question, context):
    prompt = f"""
You are a helpful sports physiology and nutrition assistant for female athletes. 
Answer the following question using ONLY the provided context.
If the answer is not contained in the context, say "I don't know based on the provided context".

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content

def evaluate_answer(client, question, answer, context):
    prompt = f"""
You are an expert evaluator assessing the quality of a RAG system.
Evaluate the generated answer based on the provided context and original question.

Question: {question}
Context: {context}
Generated Answer: {answer}

Assess the following:
1. Faithfulness: Is the generated answer entirely supported by the Context? (Score 1 if yes, 0 if no or if it contains hallucinations).
2. Relevance: Does the generated answer directly and accurately address the Question? (Score 1 if yes, 0 if no).

Provide your evaluation using the expected JSON structure.
"""
    response = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=EvaluationResult,
        temperature=0.0,
    )
    return response.choices[0].message.parsed

def main():
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Connecting to Elasticsearch...")
    es = Elasticsearch(ES_URL)
    
    print("Loading OpenAI client...")
    client = OpenAI()
    
    print("Loading chunks and ground truth...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_dict = {c["chunk_id"]: c["content"] for c in chunks}
    
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    print(f"Sampling {SAMPLE_SIZE} questions from ground truth (Seed: 42)...")
    random.seed(42)
    sample_gt = random.sample(ground_truth, SAMPLE_SIZE)
    
    results = []
    
    for item in tqdm(sample_gt, desc="Evaluating RAG"):
        question = item["question"]
        
        # 1. Retrieve top 5 chunks using our best retrieval engine (KNN Vector Search)
        retrieved_ids = search_knn(es, question, model, top_k=5)
        
        # 2. Build context block
        context_parts = []
        for cid in retrieved_ids:
            context_parts.append(chunk_dict[cid])
        context_text = "\n\n---\n\n".join(context_parts)
        
        # 3. Generate Answer using LLM
        answer = generate_answer(client, question, context_text)
        
        # 4. Evaluate with LLM as a judge
        eval_result = evaluate_answer(client, question, answer, context_text)
        
        results.append({
            "question": question,
            "answer": answer,
            "faithfulness": eval_result.faithfulness,
            "relevance": eval_result.relevance,
            "reasoning": eval_result.reasoning
        })
        
    # Calculate aggregate metrics
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS (Sample Size: {SAMPLE_SIZE})")
    print(f"{'='*50}")
    print(f"Average Faithfulness: {avg_faithfulness:.4f} ({avg_faithfulness*100:.1f}%)")
    print(f"Average Relevance:    {avg_relevance:.4f} ({avg_relevance*100:.1f}%)")
    print(f"{'='*50}")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Detailed results saved to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    main()
