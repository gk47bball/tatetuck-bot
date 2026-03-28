"""
Biopharma AutoResearcher CLI
Usage: python main.py --ticker CRSP
"""
import argparse
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

from biopharma_agent.agents.researcher import BiopharmaResearcher


def write_markdown_report(report: dict, report_path: str):
    """Generate a comprehensive markdown research report."""
    fin = report["finance_data"]
    pos = report["pos_analysis"]
    val = report["valuation"]
    heur = report["heuristic_analysis"]
    lit = report["literature_review"]
    trials = report["trials_data"]
    signal_artifact = report.get("signal_artifact", {})
    portfolio = report.get("portfolio_recommendation", {})
    snapshot = report.get("company_snapshot", {})
    
    with open(report_path, "w") as f:
        # ── Header ──
        f.write(f"# 📊 Tatetuck vNext Research Report\n")
        f.write(f"## {report['company']} ({report['ticker']})\n\n")
        f.write(f"*Generated: {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n\n")
        f.write("---\n\n")
        
        # ── Executive Summary ──
        f.write("## Executive Summary\n\n")
        signal = val.get("signal", "N/A")
        upside = val.get("implied_upside_pct", 0)
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Signal** | {signal} |\n")
        f.write(f"| **AI Probability of Success** | {pos['probability_of_success']*100:.1f}% |\n")
        f.write(f"| **Risk-Adjusted NPV (rNPV)** | ${val['rNPV']:,.0f} |\n")
        f.write(f"| **Expected Return (90d)** | {signal_artifact.get('expected_return', 0.0)*100:+.1f}% |\n")
        f.write(f"| **Catalyst Success Prob.** | {signal_artifact.get('catalyst_success_prob', 0.0)*100:.1f}% |\n")
        f.write(f"| **Recommended Scenario** | {portfolio.get('scenario', 'watchlist only')} |\n")
        f.write(f"| **Target Weight** | {portfolio.get('target_weight', 0.0)}% |\n")
        if fin.get("marketCap"):
            f.write(f"| **Current Market Cap** | ${fin['marketCap']:,.0f} |\n")
            f.write(f"| **Implied Upside/Downside** | {upside:+.1f}% |\n")
        f.write(f"| **Estimated TAM** | ${pos['estimated_tam']:,.0f} |\n")
        f.write("\n---\n\n")
        
        # ── Financials ──
        f.write("## 1. Financial Snapshot\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        if fin.get("marketCap"):
            f.write(f"| Market Cap | ${fin['marketCap']:,.0f} |\n")
        if fin.get("enterpriseValue"):
            f.write(f"| Enterprise Value | ${fin['enterpriseValue']:,.0f} |\n")
        if fin.get("totalRevenue"):
            f.write(f"| Total Revenue | ${fin['totalRevenue']:,.0f} |\n")
        if fin.get("cash"):
            f.write(f"| Cash on Hand | ${fin['cash']:,.0f} |\n")
        if fin.get("debt"):
            f.write(f"| Total Debt | ${fin['debt']:,.0f} |\n")
        if fin.get("netIncome"):
            f.write(f"| Net Income | ${fin['netIncome']:,.0f} |\n")
        f.write("\n")
        if fin.get("description"):
            f.write(f"**Company Description:** {fin['description']}\n\n")
        f.write("---\n\n")
        
        # ── Catalyst Dashboard ──
        f.write("## 2. Catalyst Dashboard\n\n")
        f.write(f"- **Thesis Horizon:** {signal_artifact.get('thesis_horizon', '90d')}\n")
        f.write(f"- **Crowding Risk:** {signal_artifact.get('crowding_risk', 0.0):.2f}\n")
        f.write(f"- **Financing Risk:** {signal_artifact.get('financing_risk', 0.0):.2f}\n")
        risk_flags = portfolio.get("risk_flags", [])
        if risk_flags:
            f.write(f"- **Risk Flags:** {', '.join(risk_flags)}\n")
        rationale = signal_artifact.get("rationale", [])
        for line in rationale:
            f.write(f"- {line}\n")
        f.write("\n---\n\n")

        # ── Clinical Trials ──
        f.write("## 3. Program & Clinical Pipeline\n\n")
        f.write(f"*{len(trials)} trials found on ClinicalTrials.gov*\n\n")
        if trials:
            f.write("| NCT ID | Title | Phase | Status | Conditions |\n")
            f.write("|--------|-------|-------|--------|------------|\n")
            for t in trials:
                nct = t.get("nct_id", "N/A")
                title = t.get("title", "N/A")[:60]
                phase = ", ".join(t.get("phase", [])) or "N/A"
                status = t.get("overall_status", "N/A")
                conds = ", ".join(t.get("conditions", [])[:2]) or "N/A"
                f.write(f"| {nct} | {title} | {phase} | {status} | {conds} |\n")
        f.write("\n")
        
        # Drug candidates
        if report.get("drug_names"):
            f.write(f"**Drug Candidates Identified:** {', '.join(report['drug_names'])}\n\n")
        f.write("---\n\n")
        
        # ── Heuristic POS ──
        f.write("## 4. Program Prior & Heuristic Analysis\n\n")
        f.write(f"Using industry-standard clinical phase transition probabilities (BIO/QLS/Informa):\n\n")
        f.write(f"- **Lead Trial:** {heur.get('lead_trial', 'N/A')}\n")
        f.write(f"- **Most Advanced Phase:** {heur.get('phase_label', 'N/A')}\n")
        f.write(f"- **Heuristic POS to Approval:** {heur['heuristic_pos']*100:.1f}%\n")
        f.write(f"- **Disease Area Multiplier:** {heur.get('disease_multiplier', 1.0):.2f}x\n")
        f.write(f"- **Est. Years to Market:** {heur.get('years_remaining', 'N/A')}\n\n")
        f.write(f"> {heur.get('details', '')}\n\n")
        f.write("---\n\n")
        
        # ── Literature Review ──
        f.write("## 5. Deep Literature Review (PubMed AutoResearch)\n\n")
        f.write(lit)
        f.write("\n\n---\n\n")
        
        # ── AI POS ──
        f.write("## 6. Catalyst-Aware Probability of Success\n\n")
        f.write(f"- **Final POS Estimate:** {pos['probability_of_success']*100:.1f}%\n")
        f.write(f"- **Estimated TAM:** ${pos['estimated_tam']:,.0f}\n\n")
        f.write(f"**AI Reasoning:**\n\n{pos['reasoning']}\n\n")
        f.write("---\n\n")
        
        # ── Valuation ──
        f.write("## 7. Event-Driven Valuation & Signal Artifact\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| Total Addressable Market | ${val['tam']:,.0f} |\n")
        f.write(f"| Peak Market Penetration | {val['penetration_rate']*100:.0f}% |\n")
        f.write(f"| Peak Revenue | ${val['peak_revenue']:,.0f} |\n")
        f.write(f"| Net Revenue at Peak | ${val['net_revenue_at_peak']:,.0f} |\n")
        f.write(f"| Unadjusted NPV | ${val['unadjusted_npv']:,.0f} |\n")
        f.write(f"| Probability of Success | {val['probability_of_success']*100:.1f}% |\n")
        f.write(f"| **Risk-Adjusted NPV (rNPV)** | **${val['rNPV']:,.0f}** |\n")
        if val.get("current_market_cap"):
            f.write(f"| Current Market Cap | ${val['current_market_cap']:,.0f} |\n")
            f.write(f"| **Implied Upside** | **{val['implied_upside_pct']:+.1f}%** |\n")
            f.write(f"| **Signal** | **{val['signal']}** |\n")
        f.write("\n")

        supporting_evidence = signal_artifact.get("supporting_evidence", [])
        if supporting_evidence:
            f.write("### Persisted Evidence Snippets\n\n")
            for item in supporting_evidence[:5]:
                f.write(f"- **{item.get('source', 'source')}**: {item.get('title', 'Evidence')} — {item.get('excerpt', '')}\n")
            f.write("\n")

        if snapshot:
            f.write("### Stored Snapshot Metadata\n\n")
            f.write(f"- **As Of:** {snapshot.get('as_of', 'N/A')}\n")
            f.write(f"- **Programs:** {len(snapshot.get('programs', []))}\n")
            f.write(f"- **Catalysts:** {len(snapshot.get('catalyst_events', []))}\n")
            f.write(f"- **Approved Products:** {len(snapshot.get('approved_products', []))}\n\n")

        # ── Footer ──
        f.write("---\n\n")
        f.write("*This report was generated by the Tatetuck vNext platform. ")
        f.write("Data sources: ClinicalTrials.gov, openFDA, PubMed (NIH), Yahoo Finance, and the local Tatetuck research store. ")
        f.write("AI analysis is used as a feature-extraction aid and stored alongside structured evidence. This is not financial advice.*\n")

    print(f"\n📄 Full report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Biopharma AutoResearcher — Karpathy-Inspired Deep Research for Small-Cap Biotech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker CRSP
  python main.py --ticker NTLA --name "Intellia Therapeutics"
  python main.py --ticker BEAM --name "Beam Therapeutics"
        """
    )
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker of the biopharma company")
    parser.add_argument("--name", type=str, help="Optional company name override for trial search")
    
    args = parser.parse_args()
    
    researcher = BiopharmaResearcher()
    report = researcher.run_research(args.ticker, args.name)
    
    report_file = os.path.join(os.path.dirname(__file__), f"report_{args.ticker.upper()}.md")
    write_markdown_report(report, report_file)


if __name__ == "__main__":
    main()
