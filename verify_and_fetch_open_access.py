"""
Athletica Capstone: Targeted Open-Access Literature Collection for Female Athletes.
Focuses strictly on Sports Nutrition, Exercise Physiology, Resistance & Endurance Training,
RED-S, Menstrual Cycle Periodization, Ergogenic Supplements, Iron/Bone Health, and Hormonal Contraception in Female Athletes (2021-2026).
"""

import os
import json
import urllib.request
import urllib.parse
import time
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Any

CATEGORIES = [
    {
        "category_id": 1,
        "category_name": "RED-S & Low Energy Availability",
        "target": 4,
        "query": '("RED-S" OR "Relative Energy Deficiency in Sport" OR "Low Energy Availability" OR "Female Athlete Triad") AND ("female athlete" OR "female athletes" OR "athletic women")'
    },
    {
        "category_id": 2,
        "category_name": "Menstrual Cycle & Nutrition/Training",
        "target": 4,
        "query": '("menstrual cycle" OR "follicular phase" OR "luteal phase") AND ("nutrition" OR "carbohydrate" OR "protein" OR "exercise performance" OR "substrate oxidation") AND ("female athlete" OR "female athletes" OR "active females")'
    },
    {
        "category_id": 3,
        "category_name": "Resistance & Endurance Training in Women",
        "target": 3,
        "query": '("resistance training" OR "muscle hypertrophy" OR "strength training" OR "endurance exercise") AND ("female athlete" OR "female athletes" OR "women athletes" OR "active women") AND ("performance" OR "adaptation")'
    },
    {
        "category_id": 4,
        "category_name": "Iron Metabolism & Bone Health in Athletes",
        "target": 3,
        "query": '("iron deficiency" OR "hepcidin" OR "ferritin" OR "iron supplementation" OR "bone mineral density" OR "stress fracture") AND ("female athlete" OR "female athletes" OR "active females")'
    },
    {
        "category_id": 5,
        "category_name": "Oral Contraceptives & Sports Performance",
        "target": 3,
        "query": '("oral contraceptives" OR "hormonal contraception") AND ("exercise performance" OR "athletic performance" OR "strength" OR "muscle" OR "recovery") AND ("female athlete" OR "active females" OR "women athletes")'
    },
    {
        "category_id": 6,
        "category_name": "Ergogenic Supplements & Sports Nutrition",
        "target": 3,
        "query": '("creatine" OR "beta-alanine" OR "caffeine" OR "ergogenic aid" OR "dietary supplement" OR "protein timing") AND ("female athlete" OR "female athletes" OR "active females" OR "women athletes")'
    }
]

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

DATA_DIR = "capstone-project/data/fulltext"
JSON_PATH = "capstone-project/pubmed_recommended_papers.json"
CSV_PATH = "capstone-project/pubmed_recommended_papers.csv"

def check_europe_pmc_xml(pmc_id: str) -> str:
    """Check if Europe PMC has XML full text for this PMC ID."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id}/fullTextXML"
    req = urllib.request.Request(url, headers={"User-Agent": "AthleticaCapstone/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                xml_str = resp.read().decode("utf-8")
                if len(xml_str) > 2000 and "</article>" in xml_str:
                    return xml_str
    except Exception:
        pass
    return ""

def parse_jats_xml_to_text(xml_str: str) -> str:
    """Parse JATS XML into clean section-structured text."""
    try:
        root = ET.fromstring(xml_str)
    except Exception:
        return ""

    text_parts = []
    
    # 1. Abstract
    abstract_nodes = root.findall(".//abstract")
    for abs_node in abstract_nodes:
        abs_text = "".join(abs_node.itertext()).strip()
        if abs_text:
            text_parts.append(f"SECTION: Abstract\n{abs_text}")

    # 2. Body sections
    body_node = root.find(".//body")
    if body_node is not None:
        for sec in body_node.findall(".//sec"):
            title_node = sec.find("title")
            title = "".join(title_node.itertext()).strip() if title_node is not None else "Section"
            paragraphs = [ "".join(p.itertext()).strip() for p in sec if p.tag == "p" and "".join(p.itertext()).strip() ]
            if paragraphs:
                sec_content = "\n".join(paragraphs)
                text_parts.append(f"SECTION: {title}\n{sec_content}")

    if not text_parts and body_node is not None:
        full_body = "".join(body_node.itertext()).strip()
        if full_body:
            text_parts.append(f"SECTION: Main Body\n{full_body}")

    return "\n\n".join(text_parts)

def search_pmc_candidates(cat: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = f'{cat["query"]} AND (Review[pt] OR "Systematic Review"[pt] OR "Consensus Development Conference"[pt]) AND "free full text"[sb]'
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 35,
        "mindate": "2021",
        "maxdate": "2026",
        "datetype": "pdat"
    }
    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Athletica/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        pmids = data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        return []

    surl = f"{ESUMMARY_URL}?db=pubmed&id={','.join(pmids)}&retmode=json"
    req_s = urllib.request.Request(surl, headers={"User-Agent": "Athletica/1.0"})
    with urllib.request.urlopen(req_s) as resp_s:
        sdata = json.loads(resp_s.read().decode('utf-8')).get("result", {})

    valid_papers = []
    for pmid in pmids:
        item = sdata.get(pmid, {})
        pmc_id = None
        doi = None
        for aid in item.get("articleids", []):
            if aid.get("idtype") == "pmc":
                pmc_id = aid.get("value")
            elif aid.get("idtype") == "doi":
                doi = aid.get("value")
        
        if pmc_id:
            xml_str = check_europe_pmc_xml(pmc_id)
            if xml_str:
                title = item.get("title", "")
                authors_list = [a.get("name") for a in item.get("authors", []) if "name" in a]
                authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
                valid_papers.append({
                    "Category ID": cat["category_id"],
                    "Category Name": cat["category_name"],
                    "PMID": pmid,
                    "PMC Free Fulltext": pmc_id,
                    "Title": title,
                    "Authors": authors_str,
                    "Publication Date": item.get("pubdate", ""),
                    "Journal": item.get("source", ""),
                    "DOI": doi or "N/A",
                    "xml_str": xml_str
                })
                if len(valid_papers) >= cat["target"]:
                    break
        time.sleep(0.3)
    return valid_papers

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clear old fulltext directory to remove non-sports/clinical files
    for existing_file in os.listdir(DATA_DIR):
        file_p = os.path.join(DATA_DIR, existing_file)
        if os.path.isfile(file_p):
            os.remove(file_p)
            
    all_final_papers = []
    seen_pmids = set()

    print("Searching PubMed & Europe PMC for 20 Sports Science & Female Athlete specific full-text papers...")

    for cat in CATEGORIES:
        print(f"\nProcessing Category {cat['category_id']}: {cat['category_name']} (Target: {cat['target']})...")
        papers = search_pmc_candidates(cat)
        for p in papers:
            if p["PMID"] not in seen_pmids:
                seen_pmids.add(p["PMID"])
                
                pmid = p["PMID"]
                pmc_id = p["PMC Free Fulltext"]
                xml_path = os.path.join(DATA_DIR, f"{pmid}_{pmc_id}.xml")
                txt_path = os.path.join(DATA_DIR, f"{pmid}_{pmc_id}.txt")
                
                clean_text = parse_jats_xml_to_text(p["xml_str"])
                with open(xml_path, "w", encoding="utf-8") as fx:
                    fx.write(p["xml_str"])
                with open(txt_path, "w", encoding="utf-8") as ft:
                    ft.write(clean_text)
                    
                print(f"  [OK] Saved {pmc_id} (PMID {pmid}): {p['Title'][:55]}... ({len(clean_text)} chars)")
                
                meta_item = dict(p)
                del meta_item["xml_str"]
                meta_item["PubMed Link"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                meta_item["PMC / Fulltext Link"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
                all_final_papers.append(meta_item)

    print(f"\n=========================================================")
    print(f" TOTAL VERIFIED ATHLETE-SPECIFIC OPEN ACCESS PAPERS: {len(all_final_papers)}")
    print(f"=========================================================")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_final_papers, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(all_final_papers)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()
