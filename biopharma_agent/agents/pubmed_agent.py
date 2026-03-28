"""
AutoResearch Agent: Karpathy-style iterative literature synthesis.

Searches PubMed for relevant scientific papers, then uses an LLM (if available)
to write a deep-dive literature review. Falls back to a structured abstract
summary if no LLM key is present.
"""
import os
import time
from typing import List, Dict, Any
from ..data.pubmed import PubMedAPI

class AutoResearchAgent:
    def __init__(self):
        self.pubmed = PubMedAPI()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        self.model_name = 'gemini-2.5-flash'
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client()
            except Exception:
                self.client = None

    @property
    def has_llm(self) -> bool:
        return self.client is not None

    def _build_search_queries(self, company_name: str, drug_names: List[str], conditions: List[str]) -> List[str]:
        """Build multiple targeted PubMed queries for iterative search."""
        queries = []
        # Query 1: Company + clinical trial
        queries.append(f'"{company_name}"[All Fields] AND clinical trial[pt]')
        # Query 2: Each drug name
        for drug in drug_names[:3]:
            queries.append(f'"{drug}"[All Fields] AND (efficacy OR safety)')
        # Query 3: Conditions + mechanism
        for cond in conditions[:2]:
            queries.append(f'"{cond}"[All Fields] AND (treatment OR therapy) AND clinical trial[pt]')
        return queries

    def generate_literature_review(self, company_name: str, drug_names: List[str], conditions: List[str] = None) -> str:
        """
        Karpathy-style AutoResearch:
        1. Build multiple search queries
        2. Iteratively search PubMed
        3. Deduplicate and rank papers
        4. Synthesize with LLM (or format as structured summary)
        """
        conditions = conditions or []
        queries = self._build_search_queries(company_name, drug_names, conditions)
        
        all_papers = {}
        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(0.5)  # NIH E-utilities: max 3 requests/sec without API key
            print(f"  [AutoResearch] Searching PubMed: {query[:80]}...")
            try:
                papers = self.pubmed.search_abstracts(query, max_results=5)
                for p in papers:
                    if p.get("pmid") and p["pmid"] not in all_papers:
                        all_papers[p["pmid"]] = p
            except Exception as e:
                print(f"  [AutoResearch] Search failed: {e}")

        papers_list = list(all_papers.values())
        print(f"  [AutoResearch] Total unique papers found: {len(papers_list)}")

        if not papers_list:
            return "No relevant scientific literature found in PubMed for this company or its drug candidates."

        # Build the abstracts context
        abstracts_text = ""
        for i, p in enumerate(papers_list):
            abstracts_text += f"\n[{i+1}] **{p['title']}** (PMID: {p['pmid']})\n{p['abstract']}\n"

        if not self.has_llm:
            # Heuristic fallback: structured summary without LLM
            review = f"### PubMed Literature Summary ({len(papers_list)} papers found)\n\n"
            review += "The following scientific publications were identified as relevant:\n\n"
            for i, p in enumerate(papers_list):
                review += f"**[{i+1}]** {p['title']} (PMID: {p['pmid']})\n"
                # Show first 300 chars of abstract
                snippet = p['abstract'][:300] + "..." if len(p['abstract']) > 300 else p['abstract']
                review += f"> {snippet}\n\n"
            review += "\n*Full LLM synthesis requires GEMINI_API_KEY.*\n"
            return review

        # LLM-powered deep synthesis
        prompt = f"""You are a senior biopharma research analyst conducting due diligence for a healthcare-focused hedge fund.

You have been given scientific abstracts related to **{company_name}** and its drug candidates: {drug_names}.

Write a rigorous, detailed literature review. Structure it as:

## Mechanism of Action & Scientific Rationale
Explain the biological mechanism being targeted. Why is this approach promising?

## Clinical Efficacy Evidence
What do the trial results show? What are the key endpoints and how strong is the data?

## Safety Profile & Risks
Any safety concerns from the literature? Adverse events? Competitive risks?

## Investment Implications
What does the science tell us about the probability of regulatory success? Any alpha signals?

Cite papers using [1], [2], etc. Be specific about data points, p-values, response rates where available.

### Source Abstracts:
{abstracts_text}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={'temperature': 0.3}
            )
            return response.text
        except Exception as e:
            return f"LLM synthesis failed: {e}. Raw abstracts:\n{abstracts_text}"
