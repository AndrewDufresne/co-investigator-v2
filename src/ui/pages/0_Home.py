"""Argus AI — Streamlit Home page."""

from __future__ import annotations

import streamlit as st


def run() -> None:
    st.title("🔍 Argus AI")
    st.subheader("Agentic AI for Smarter, Trustworthy Anti-fraud and Anti-money laundering Compliance Narratives")

    st.markdown("""
Welcome to **Argus AI** — an agentic multi-agent system that automates
Suspicious Activity Report (SAR) narrative generation for anti-fraud and anti-money laundering compliance.
### 🚀 How It Works

1. **📄 Upload Case** — Upload a JSON case file with transaction data, KYC, and alerts
2. **🔍 Generate SAR** — The AI pipeline analyzes the case through 10 specialized agents
3. **✏️ Review & Refine** — Review the generated narrative, provide feedback, and iterate
4. **📊 Dashboard** — Visualize risk analysis and typology detection results

### 🧠 Agent Pipeline

| Agent | Role |
|-------|------|
| Data Ingestion | Parses and structures raw case data |
| AI-Privacy Guard | Masks PII before LLM processing |
| Crime Detection | Identifies financial crime typologies |
| Planning Agent | Orchestrates analysis strategy |
| Typology Agents (7) | Specialized pattern detection |
| External Intel | Gathers external risk intelligence |
| Narrative Generator | Produces FinCEN-compliant SAR narratives |
| Compliance Validator | Agent-as-a-Judge quality assurance |
| Feedback Loop | Human-in-the-Loop refinement |

---
*Use the sidebar to navigate between pages.*
""")


if __name__ == "__main__":
    run()
