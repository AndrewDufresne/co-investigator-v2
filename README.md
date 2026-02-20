# Co-Investigator V2

> Agentic AI framework for AML Suspicious Activity Report (SAR) narrative generation.

Based on the paper: *"Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* (arXiv:2509.08380v2).

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

Copy `.env.example` to `.env` and set your DeepSeek API key:

```bash
DEEPSEEK_API_KEY=your-key-here
```

### 3. Run the application

```bash
streamlit run src/app.py
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## Project Structure

```
src/
├── app.py              # Streamlit entry point
├── config.py           # Application configuration
├── core/               # State, models, LLM gateway
├── graph/              # LangGraph workflow definitions
├── agents/             # Agent implementations (LangGraph nodes)
├── infrastructure/     # Memory, tools, MCP client
└── ui/                 # Streamlit pages & components
```
