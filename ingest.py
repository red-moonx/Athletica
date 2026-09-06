import json
import os
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
import warnings
from dotenv import load_dotenv

# Suppress Elasticsearch warnings for cleaner output
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(".env")

# Configuration
ES_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
INDEX_NAME = "athletica_chunks"
CHUNKS_PATH = "data/processed/optimizations/optimization_3/chunks.json"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

def setup_elasticsearch(es):
    # Check if connected
    print("Connecting to ES...")
    info = es.info()
    print("Connected to Elasticsearch:", info['version']['number'])
        
    # Delete index if exists
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")
        
    # Create index with mapping including dense_vector
    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "pmid": {"type": "keyword"},
                "title": {"type": "text"},
                "authors": {"type": "text"},
                "category_name": {"type": "keyword"},
                "publication_date": {"type": "keyword"},
                "section_id": {"type": "keyword"},
                "content": {"type": "text"},
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    es.indices.create(index=INDEX_NAME, mappings=mapping["mappings"])
    print(f"Created index: {INDEX_NAME}")

def index_data(es, chunks, model):
    print(f"Generating embeddings for all chunks using {MODEL_NAME}...")
    contents = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(contents, show_progress_bar=True)
    
    print(f"Indexing {len(chunks)} chunks into {INDEX_NAME}...")
    actions = []
    for i, chunk in enumerate(chunks):
        action = {
            "_index": INDEX_NAME,
            "_id": chunk["chunk_id"],
            "_source": {
                "chunk_id": chunk["chunk_id"],
                "pmid": chunk["pmid"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "category_name": chunk["category_name"],
                "publication_date": chunk["publication_date"],
                "section_id": chunk.get("section_id", ""),
                "content": chunk["content"],
                "dense_vector": embeddings[i].tolist()
            }
        }
        actions.append(action)
        
    success, failed = bulk(es, actions)
    print(f"Successfully indexed {success} documents.")
    if failed:
        print(f"Failed to index {len(failed)} documents.")
    
    # Force refresh index to make documents immediately searchable
    es.indices.refresh(index=INDEX_NAME)

def main():
    parser = argparse.ArgumentParser(description="Ingest chunks into Elasticsearch")
    parser.add_argument("--chunks", default=CHUNKS_PATH, help="Path to chunks.json")
    args = parser.parse_args()

    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    es = Elasticsearch(ES_URL, request_timeout=120)
    setup_elasticsearch(es)
    
    print(f"Loading chunks from {args.chunks}...")
    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    index_data(es, chunks, model)

if __name__ == "__main__":
    main()
