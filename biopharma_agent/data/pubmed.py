from typing import List, Dict, Any

from prepare import fetch_pubmed_abstracts

class PubMedAPI:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search_abstracts(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Delegate to prepare.py so PubMed calls share the evaluator's cache and parsing."""
        return fetch_pubmed_abstracts(query, max_results=max_results)
