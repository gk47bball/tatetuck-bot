import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

class PubMedAPI:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search_abstracts(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Search PubMed and retrieve abstracts for a given query (Free NIH E-utilities)."""
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []
            
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_resp = requests.get(fetch_url, params=fetch_params)
        fetch_resp.raise_for_status()
        
        root = ET.fromstring(fetch_resp.content)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            abstract_texts = article.findall(".//AbstractText")
            abstract = " ".join([elem.text for elem in abstract_texts if elem.text])
            
            if title and abstract:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract
                })
                
        return list({v['pmid']:v for v in articles}.values()) # Deduplicate
