import json
import numpy as np
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer, CrossEncoder
import warnings

# Suppress Elasticsearch warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration
ES_URL = "http://localhost:9200"
INDEX_NAME = "athletica_chunks"
CHUNKS_PATH = "data/processed/optimizations/optimization_3/chunks.json"
GROUND_TRUTH_PATH = "data/processed/optimizations/optimization_3/ground_truth_round_3.json"
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

def setup_elasticsearch():
    es = Elasticsearch(ES_URL, request_timeout=120)
    
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
    return es

def index_data(es, chunks, model):
    print("Generating embeddings for all chunks...")
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

def search_text(es, query: str, top_k: int = 5):
    search_body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content^3", "title", "category_name"],
                "type": "best_fields"
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body)
    return [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]

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

def search_hybrid(es, query: str, model, top_k: int = 5):
    vector = model.encode(query).tolist()
    search_body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["content^3", "title", "category_name"],
                "type": "best_fields",
                "boost": 0.5
            }
        },
        "knn": {
            "field": "dense_vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": 100,
            "boost": 0.5
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body)
    return [hit["_source"]["chunk_id"] for hit in response["hits"]["hits"]]

def search_knn_reranked(es, query: str, model, cross_encoder, chunk_dict, top_k: int = 5):
    # Retrieve top 50 candidates using KNN
    knn_results = search_knn(es, query, model, top_k=50)
    
    # Create pairs of (query, document_content)
    pairs = [[query, chunk_dict[chunk_id]] for chunk_id in knn_results]
    
    # Predict relevance scores
    scores = cross_encoder.predict(pairs)
    
    # Combine chunk_ids and scores, then sort descending
    results = list(zip(knn_results, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top_k
    return [x[0] for x in results[:top_k]]

def evaluate(ground_truth, search_func, k_values=[5, 10, 15]):
    max_k = max(k_values)
    results = {k: {'hits': [], 'reciprocal_ranks': []} for k in k_values}
    
    for item in tqdm(ground_truth, desc="Evaluating"):
        query = item["question"]
        expected_id = item["chunk_id"]
        
        retrieved_ids = search_func(query, top_k=max_k)
        
        for k in k_values:
            k_retrieved_ids = retrieved_ids[:k]
            if expected_id in k_retrieved_ids:
                results[k]['hits'].append(1)
                rank = k_retrieved_ids.index(expected_id) + 1
                results[k]['reciprocal_ranks'].append(1.0 / rank)
            else:
                results[k]['hits'].append(0)
                results[k]['reciprocal_ranks'].append(0.0)
            
    metrics = {}
    for k in k_values:
        hit_rate = np.mean(results[k]['hits'])
        mrr = np.mean(results[k]['reciprocal_ranks'])
        metrics[k] = {'hit_rate': hit_rate, 'mrr': mrr}
        
    return metrics

def print_metrics(name, metrics):
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({name})")
    print(f"{'='*50}")
    for k, vals in metrics.items():
        print(f"Metrics @ {k}:")
        print(f"  Hit Rate: {vals['hit_rate']:.4f} ({vals['hit_rate']*100:.2f}%)")
        print(f"  MRR:      {vals['mrr']:.4f}")
        print("-" * 50)

def main():
    print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    es = setup_elasticsearch()
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Create chunk_dict for fast lookup during reranking
    chunk_dict = {c["chunk_id"]: c["content"] for c in chunks}
    
    # Skipping indexing since it's already done in the previous run
    index_data(es, chunks, model)
    
    # 2. Evaluate
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    print("\n--- Evaluating Text Search (BM25) ---")
    metrics_text = evaluate(ground_truth, lambda q, top_k=5: search_text(es, q, top_k))
    
    print("\n--- Evaluating KNN Search (Vector) ---")
    metrics_knn = evaluate(ground_truth, lambda q, top_k=5: search_knn(es, q, model, top_k))
    
    print("\n--- Evaluating Hybrid Search (BM25 + KNN) ---")
    metrics_hybrid = evaluate(ground_truth, lambda q, top_k=5: search_hybrid(es, q, model, top_k))
    
    print("\n--- Evaluating KNN Search + CrossEncoder Reranking ---")
    metrics_reranked = evaluate(ground_truth, lambda q, top_k=5: search_knn_reranked(es, q, model, cross_encoder, chunk_dict, top_k))
    
    # Print results
    print_metrics("Text Search (BM25)", metrics_text)
    print_metrics("KNN Search (Vector)", metrics_knn)
    print_metrics("Hybrid Search (BM25 + KNN)", metrics_hybrid)
    print_metrics("KNN Search + Reranker", metrics_reranked)

if __name__ == "__main__":
    main()
