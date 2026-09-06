"""
Section-aware Parser & Chunker for Athletica Capstone RAG.
Reads full text files from Europe PMC (or PDFs/metadata) and PubMed JSON to produce structured chunks.
"""

import os
import json
import re
import pandas as pd
import pypdf
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

FULLTEXT_DIR = "capstone-project/data/fulltext"
PDF_DIR = "capstone-project/data/pdfs"
JSON_PATH = "capstone-project/pubmed_recommended_papers.json"
OUTPUT_DIR = "capstone-project/data/processed"

def extract_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def split_text_into_chunks(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[Dict[str, str]]:
    """Split text into overlapping chunks by sections, filtering out references."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    sections = []
    if "SECTION:" in text:
        # Split by section marker
        raw_sections = re.split(r'\n?SECTION:\s*', text)
        for s in raw_sections:
            s_clean = s.strip()
            if not s_clean:
                continue
            
            parts = s_clean.split('\n', 1)
            section_title = parts[0].strip()
            section_content = parts[1].strip() if len(parts) > 1 else ""
            
            # Filter garbage sections
            title_lower = section_title.lower()
            garbage_keywords = ['reference', 'bibliography', 'acknowledgement', 'declaration', 'funding', 'data availability', 'author contribution', 'abbreviation']
            if any(k in title_lower for k in garbage_keywords):
                continue
                
            if len(section_content) > 50:
                sections.append((section_title, section_content))
    else:
        # Fallback if no sections
        sections = [("General", text)]
        
    chunks_with_meta = []
    for section_title, section_content in sections:
        texts = text_splitter.split_text(section_content)
        for t in texts:
            if len(t.strip()) > 50:
                chunks_with_meta.append({
                    "section_id": section_title,
                    "content": t.strip()
                })
            
    return chunks_with_meta

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    print(f"Processing {len(papers)} candidate papers for chunking...")
    all_chunks = []
    
    for paper in papers:
        pmid = paper["PMID"]
        pmc_id = paper.get("PMC Free Fulltext")
        title = paper.get("Title")
        authors = paper.get("Authors")
        category = paper.get("Category Name")
        pubdate = paper.get("Publication Date")
        
        txt_path = os.path.join(FULLTEXT_DIR, f"{pmid}_{pmc_id}.txt") if pmc_id and pmc_id.startswith("PMC") else None
        pdf_path = os.path.join(PDF_DIR, f"{pmid}_{pmc_id}.pdf") if pmc_id and pmc_id.startswith("PMC") else None
        
        full_text = ""
        source_type = "metadata"
        
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as tf:
                full_text = tf.read()
            if len(full_text.strip()) > 300:
                source_type = "fulltext_xml"
                
        if source_type == "metadata" and pdf_path and os.path.exists(pdf_path):
            pdf_text = extract_pdf_text(pdf_path)
            if len(pdf_text.strip()) > 500 and "Preparing to download" not in pdf_text:
                full_text = pdf_text
                source_type = "pdf"
                
        if source_type == "metadata":
            # Fallback to abstract & metadata
            abstract = paper.get("Abstract", "")
            full_text = f"Title: {title}\nAuthors: {authors}\nCategory: {category}\nPublication Date: {pubdate}\nJournal: {paper.get('Journal')}\nPMID: {pmid}\n\nAbstract:\n{abstract}"
            
        raw_chunks = split_text_into_chunks(full_text, chunk_size=2000, overlap=400)
        
        for idx, chunk_dict in enumerate(raw_chunks, start=1):
            chunk_str = chunk_dict["content"]
            section_id = chunk_dict["section_id"]
            chunk_id = f"PMID_{pmid}_c{idx:02d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "category_name": category,
                "publication_date": pubdate,
                "source_type": source_type,
                "section_id": section_id,
                "chunk_index": idx,
                "content": chunk_str,
                "word_count": len(chunk_str.split()),
                "char_count": len(chunk_str)
            })
            
    print(f"\nGenerated total {len(all_chunks)} chunks across {len(papers)} papers.")
    
    # Save outputs
    json_out = os.path.join(OUTPUT_DIR, "chunks.json")
    csv_out = os.path.join(OUTPUT_DIR, "chunks.csv")
    
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    df = pd.DataFrame(all_chunks)
    df.to_csv(csv_out, index=False, encoding="utf-8")
    
    print(f"Exported chunks JSON to {json_out}")
    print(f"Exported chunks CSV to {csv_out}")

if __name__ == "__main__":
    main()
