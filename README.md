# 🔍 Argus V2

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Agentic AI framework for AML Suspicious Activity Report (SAR) narrative generation.**

> 💡 **LLM Compatibility:** This project is compatible with any LLM service that follows the OpenAI API format (e.g., DeepSeek, OpenAI, Azure OpenAI, Ollama, etc.). DeepSeek is used as an example throughout the documentation.

Based on the paper: *"Argus AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* (arXiv:2509.08380v2).

**🌐 [中文版 README](README_zh.md)**

---

## ✨ Key Features

- **Multi-Agent Architecture** — SAR generation decomposed into specialized agents (ingestion, crime detection, planning, typology analysis, narrative generation, compliance validation), orchestrated via LangGraph state graphs
- **Human-in-the-Loop** — Investigators review and edit AI-generated drafts before submission; feedback triggers iterative refinement
- **Privacy-First Design** — AI-Privacy Guard anonymizes sensitive data (PII) before LLM processing and restores it afterward
- **Dynamic Typology Analysis** — 7 specialized typology agents (transaction fraud, payment velocity, country risk, text content, geo anomaly, account health, dispute pattern) activated dynamically based on detected crime types
- **Compliance Validation** — Agent-as-a-Judge pattern with dual rule-based and semantic validation to ensure FinCEN compliance
- **Chain-of-Thought Explainability** — Full CoT reasoning traces for transparency and auditability
- **Three-Layer Dynamic Memory** — Regulatory memory, historical narrative memory, and typology-specific memory for context-augmented generation

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Application                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 UI Layer (5 Pages)                    │  │
│  │   📄 Case Upload → 🔍 SAR Generate → ✏️ Review      │  │
│  │  📊 Dashboard → 📋 History                           │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │           LangGraph Orchestration Layer               │  │
│  │                                                       │  │
│  │  [ingest] → [privacy_mask] → [crime_detect] → [plan]  │  │
│  │    → [typology_subgraph] → [external_intel]           │  │
│  │    → [narrative_generate] → [compliance_validate]     │  │
│  │    → PASS: [privacy_unmask] → END                     │  │
│  │    → FAIL: [feedback_refine] → (iterate)              │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │              Infrastructure Layer                     │  │
│  │  Privacy Guard │ LLM Gateway │ Dynamic Memory │ MCP   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

For the full architecture design, see:
- 📖 [Architecture Document (English)](docs/Architecture_en.md)
- 📖 [架构设计文档 (中文)](docs/Architecture_zh.md)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An API key for any OpenAI API-compatible LLM service ([DeepSeek](https://platform.deepseek.com/) is used as an example below)

### 1. Clone and install

```bash
git clone https://github.com/your-org/argus-v2.git
cd argus-v2
pip install -e .
```

### 2. Configure environment

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
```

```env
LLM_API_KEY=your-key-here
```

**Available configuration options:**

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | LLM API key (required; compatible with any OpenAI API-format service) |
| `LLM_BASE_URL` | `https://api.deepseek.com` | LLM API base URL (replace with your service endpoint) |
| `LLM_MODEL` | `deepseek-chat/deepseek-reasoner` | Model name (replace with your model name) |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature (low for SAR precision) |
| `COMPLIANCE_SCORE_THRESHOLD` | `0.75` | Minimum compliance score to pass |
| `MAX_ITERATIONS` | `3` | Maximum feedback iteration rounds |

### 3. Run the application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`.

---

## 📁 Project Structure

```
Argus-v2/
├── src/
│   ├── app.py                  # Streamlit application entry point
│   ├── config.py               # Pydantic Settings configuration
│   ├── core/                   # Core abstractions
│   │   ├── state.py            #   SARState TypedDict definition
│   │   ├── models.py           #   Pydantic data models
│   │   └── llm_gateway.py      #   Unified LLM Gateway (DeepSeek-first)
│   ├── graph/                  # LangGraph workflow definitions
│   │   ├── sar_graph.py        #   Main graph: SARGenerationGraph
│   │   ├── typology_subgraph.py#   Subgraph: TypologySubgraph
│   │   └── routing.py          #   Conditional routing functions
│   ├── agents/                 # Agent implementations (LangGraph nodes)
│   │   ├── ingestion.py        #   Data Ingestion Agent
│   │   ├── privacy_guard.py    #   AI-Privacy Guard (mask/unmask)
│   │   ├── crime_detection.py  #   Crime Type Detection Agent
│   │   ├── planning.py         #   Planning Agent (orchestrator)
│   │   ├── narrative.py        #   Narrative Generation Agent
│   │   ├── compliance.py       #   Compliance Validation (Agent-as-a-Judge)
│   │   ├── feedback.py         #   Feedback Agent
│   │   ├── external_intel.py   #   External Intelligence Agent (MCP)
│   │   └── typology/           #   7 Specialized Typology Agents
│   ├── infrastructure/         # Infrastructure layer
│   │   └── memory/             #   Three-layer dynamic memory
│   └── ui/                     # Streamlit UI layer
│       ├── pages/              #   Multi-page app (5 pages)
│       ├── components/         #   Reusable UI components
│       └── session.py          #   Session State management
├── data/samples/               # Sample case data (JSON)
├── prompts/                    # Prompt templates (YAML)
├── evaluation/                 # Evaluation framework
├── tests/                      # Unit & integration tests
├── docs/                       # Architecture documentation
├── pyproject.toml              # Project dependencies & metadata
└── README.md
```

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run smoke test
pytest tests/smoke_test.py

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/
```

---

## 🔧 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent Orchestration** | LangGraph v0.2+ | Stateful graph execution, conditional routing, HITL interrupts |
| **LLM** | OpenAI API-compatible services (via LangChain; DeepSeek as example) | Primary LLM for narrative generation, planning, compliance |
| **UI** | Streamlit | Interactive investigator interface with real-time streaming |
| **Crime Detection** | scikit-learn (RF/GBM) | Tree-based ensemble crime type classification |
| **Privacy** | Presidio / spaCy NER | PII detection and anonymization (MVP) |
| **Memory** | ChromaDB + SQLite | Vector search + structured storage |
| **External Intel** | MCP Client SDK | Dynamic external data source integration |
| **Visualization** | Plotly | Risk charts, transaction timelines, heatmaps |

---

## 📋 Implementation Roadmap

| Phase | Objective | Status |
|---|---|---|
| **Phase 0** | Project skeleton — structure, config, state, LLM Gateway | ✅ Complete |
| **Phase 1** | Minimal pipeline — Ingest → Crime Detect → Narrative → Output | ✅ Complete |
| **Phase 2** | Full agent suite — Planning, 7 Typology Agents, Compliance, Feedback Loop | ✅ Complete |
| **Phase 3** | Security & memory — Privacy Guard, 3-layer memory, External Intel (MCP) | 🔄 In Progress |
| **Phase 4** | Human-AI collaboration — HITL interrupts, review pages, feedback UI | ⬚ Planned |
| **Phase 5** | Evaluation & optimization — Offline eval, Agent-as-a-Judge, prompt tuning | ⬚ Planned |

---

## 📄 License

This project is for research and educational purposes.

---

## 📚 References

- Paper: *"Argus AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* — [arXiv:2509.08380v2](https://arxiv.org/abs/2509.08380v2)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DeepSeek API](https://platform.deepseek.com/)
