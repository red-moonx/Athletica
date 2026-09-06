import json
import os
import random
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

# Load environment variables
load_dotenv()

class QAPair(BaseModel):
    question: str = Field(description="A natural, colloquial, 'down to earth' question that a recreational female athlete would ask, based on the chunk.")
    answer: str = Field(description="The factual answer based strictly on the provided text chunk.")

class QAList(BaseModel):
    qa_pairs: List[QAPair]

def generate_qa_for_chunk(client, chunk_content: str, chunk_id: str) -> List[dict]:
    prompt = f"""
You are a normal, everyday active woman (e.g., a 30-year-old recreational gym-goer or runner) using a sports health app. 
Read the following scientific text about female physiology/nutrition. 
Based ONLY on this text, generate 2 realistic, conversational, and practical 'down to earth' questions that YOU would ask the app. 
Example: Instead of asking 'What are the metabolic effects of LEA?', ask 'I am a 30-year-old woman and I feel like I lack energy in the mornings for my workouts. What can I do?'

Text Chunk:
{chunk_content}
    """

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates synthetic Q&A data based on scientific texts."},
                {"role": "user", "content": prompt}
            ],
            response_format=QAList,
            temperature=0.7,
        )
        
        result = []
        for pair in completion.choices[0].message.parsed.qa_pairs:
            result.append({
                "chunk_id": chunk_id,
                "question": pair.question,
                "answer": pair.answer
            })
        return result
    except Exception as e:
        print(f"Error generating QA for chunk {chunk_id}: {e}")
        return []

def main():
    print("Loading chunks...")
    input_path = "../../data/processed/chunks.json"
    output_path = "ground_truth_round_1.json"
    
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Pick 300 random chunks for the final evaluation dataset
    random.seed(42) # Fixed seed for reproducibility
    sample_chunks = random.sample(chunks, 300)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    all_qa_data = []
    
    print(f"Generating questions for {len(sample_chunks)} sample chunks...")
    for chunk in sample_chunks:
        print(f"Processing {chunk['chunk_id']} ({chunk['category_name']})")
        qa_pairs = generate_qa_for_chunk(client, chunk["content"], chunk["chunk_id"])
        all_qa_data.extend(qa_pairs)
        
        # Print to console for immediate verification
        for qa in qa_pairs:
            print(f"\nQ: {qa['question']}")
            print(f"A: {qa['answer']}")
            print("-" * 40)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nSaved {len(all_qa_data)} test Q&A pairs to {output_path}")

if __name__ == "__main__":
    main()
