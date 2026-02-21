# Argus V2 — System Architecture Design Document

> Implementation based on the methodology from the paper *"Argus AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* (arXiv:2509.08380v2).

---

## 1. Architecture Overview

### 1.1 Design Principles

| Principle | Description |
|---|---|
| **Agentic Decomposition** | Decompose SAR generation into multiple specialized Agents, each with a single responsibility and independently evolvable |
| **LangGraph-Driven** | All Agent orchestration is based on LangGraph state graphs, enabling controllable and observable workflows |
| **Human-in-the-Loop** | Human investigators are always in the loop; AI generates drafts for review, never auto-submits |
| **Privacy-First** | Sensitive data must be anonymized by the AI-Privacy Guard before being sent to any LLM |
| **Monolithic Deployment** | Single-process Streamlit application with internal LangGraph orchestration, reducing operational complexity |

### 1.2 Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Most mature ecosystem for Agent / LLM / NLP development |
| **Agent Orchestration** | LangGraph (v0.2+) | Provides stateful graph execution, conditional routing, human-in-the-loop interrupts, and checkpoint persistence |
| **LLM** | DeepSeek (Primary) | Primary support; extensible to other models via a unified Gateway |
| **LLM Integration** | LangChain ChatModel Abstraction | Native LangGraph support, unified interface for DeepSeek / OpenAI / Anthropic |
| **Privacy Layer** | RoBERTa + CRF (Paper Design) | MVP stage uses presidio / spaCy NER first, later replaced with self-trained model |
| **Crime Type Detection** | scikit-learn (RF / GBM) | Tree-based ensemble methods as specified in the paper |
| **Memory Layer** | ChromaDB (Vector) + SQLite (Structured) | Lightweight, suitable for monolithic deployment |
| **UI Framework** | Streamlit | Rapid data application UI development with native support for interactive widgets, real-time streaming, and Session State management |
| **Data Format** | JSON | Unified JSON for input / output / sample data |
| **Configuration** | Pydantic Settings + YAML | Type-safe configuration loading |

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Application                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   UI Layer (Streamlit)                     │  │
│  │                                                           │  │
│  │  📄 Case Upload    — JSON file upload / sample selection   │  │
│  │  🔍 SAR Generate   — One-click trigger, real-time stream   │  │
│  │  ✏️ Narrative Review — Draft display + inline edit + feedback│ │
│  │  📊 Analysis Dashboard — Crime types, risk, compliance viz │  │
│  │  📋 History        — Historical SAR list + search + export │  │
│  │                                                           │  │
│  │  State Mgmt: st.session_state (case data/graph state/feedback)│
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              LangGraph Orchestration Layer                 │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │           SARGenerationGraph (Main Graph)           │  │  │
│  │  │                                                     │  │  │
│  │  │  [ingest] → [privacy_mask] → [crime_detect]         │  │  │
│  │  │      → [plan] → [typology_subgraph] ──┐             │  │  │
│  │  │                                       ▼             │  │  │
│  │  │      [external_intel] → [narrative_generate]        │  │  │
│  │  │           → [compliance_validate] ──┐               │  │  │
│  │  │                                     ▼               │  │  │
│  │  │  ┌──── PASS ──── [privacy_unmask] → [human_review]  │  │  │
│  │  │  │                                       │          │  │  │
│  │  │  │    FAIL ── [feedback_refine] ─────────┘          │  │  │
│  │  │  │                  ▲        │                       │  │  │
│  │  │  │                  └────────┘ (iteration)           │  │  │
│  │  └──┴──────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │      TypologySubgraph (Subgraph, Dynamic Parallel)  │  │  │
│  │  │                                                     │  │  │
│  │  │  [transaction_fraud]     [payment_velocity]         │  │  │
│  │  │  [country_risk]          [text_content]             │  │  │
│  │  │  [geo_anomaly]           [account_health]           │  │  │
│  │  │  [dispute_pattern]                                  │  │  │
│  │  │        ──── all converge → [typology_merge] ────    │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │                 Infrastructure Layer                      │  │
│  │                                                           │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────────┐ │  │
│  │  │ Privacy    │ │ LLM        │ │ Dynamic Memory         │ │  │
│  │  │ Guard      │ │ Gateway    │ │ ┌────────┐ ┌────────┐  │ │  │
│  │  │ (NER+CRF)  │ │ (DeepSeek) │ │ │Reg.Mem │ │Hist.Mem│  │ │  │
│  │  │            │ │            │ │ │(Chroma)│ │(Chroma)│  │ │  │
│  │  │            │ │            │ │ ├────────┤ ├────────┤  │ │  │
│  │  │            │ │            │ │ │Typo.Mem│ │State   │  │ │  │
│  │  │            │ │            │ │ │(SQLite)│ │(SQLite)│  │ │  │
│  │  └────────────┘ └────────────┘ │ └────────┘ └────────┘  │ │  │
│  │                                └────────────────────────┘ │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────────┐ │  │
│  │  │ Data       │ │ Analytical │ │ MCP Client             │ │  │
│  │  │ Ingestion  │ │ Tools      │ │ (External Intel)       │ │  │
│  │  └────────────┘ └────────────┘ └────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. LangGraph Core Design

### 3.1 Global State (State Schema)

LangGraph uses a **TypedDict** to define the shared state that flows through the entire workflow. All Agents (nodes) read from and write to the same State object:

```python
class SARState(TypedDict):
    # ── Input Data ──
    case_id: str                          # Unique case identifier
    raw_data: dict                        # Raw JSON case data
    structured_data: dict                 # Structured/normalized data
    masked_data: dict                     # Anonymized data
    mask_mapping: dict                    # Anonymization mapping table (for de-masking)

    # ── Crime Type Detection ──
    risk_indicators: list[dict]           # Extracted risk indicators
    crime_types: list[CrimeTypeResult]    # Detected crime types + confidence scores

    # ── Planning ──
    execution_plan: ExecutionPlan         # Execution plan generated by the Planning Agent
    active_typology_agents: list[str]     # List of typology agents to activate

    # ── Typology Detection Results ──
    typology_results: dict[str, dict]     # Analysis results from each typology agent

    # ── External Intelligence ──
    external_intel: list[dict]            # External intelligence retrieved via MCP

    # ── Narrative Generation ──
    narrative_draft: str                  # Current narrative draft
    narrative_intro: str                  # Narrative introduction section
    chain_of_thought: list[str]           # Chain-of-Thought reasoning trace

    # ── Compliance Validation ──
    compliance_result: ComplianceResult   # Validation result (PASS/FAIL + details)
    compliance_score: float               # Compliance score

    # ── Human-in-the-Loop ──
    human_feedback: str | None            # Human feedback content
    iteration_count: int                  # Current iteration round
    max_iterations: int                   # Maximum iteration count

    # ── Final Output ──
    final_narrative: str                  # Final SAR narrative
    status: Literal["processing", "review", "approved", "rejected"]
    messages: Annotated[list, add_messages]  # Inter-agent message log
```

### 3.2 Main Graph (SARGenerationGraph)

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(SARState)

# ── Register Nodes (each node corresponds to an Agent function) ──
graph.add_node("ingest",              data_ingestion_agent)
graph.add_node("privacy_mask",        privacy_mask_agent)
graph.add_node("crime_detect",        crime_detection_agent)
graph.add_node("plan",                planning_agent)
graph.add_node("typology_analysis",   typology_subgraph)      # Subgraph
graph.add_node("external_intel",      external_intel_agent)
graph.add_node("narrative_generate",  narrative_generation_agent)
graph.add_node("compliance_validate", compliance_validation_agent)
graph.add_node("privacy_unmask",      privacy_unmask_agent)
graph.add_node("feedback_refine",     feedback_agent)

# ── Define Edges (linear + conditional routing) ──
graph.add_edge(START,                 "ingest")
graph.add_edge("ingest",             "privacy_mask")
graph.add_edge("privacy_mask",       "crime_detect")
graph.add_edge("crime_detect",       "plan")
graph.add_edge("plan",               "typology_analysis")
graph.add_edge("typology_analysis",  "external_intel")
graph.add_edge("external_intel",     "narrative_generate")
graph.add_edge("narrative_generate", "compliance_validate")

# Conditional routing: compliance pass → unmask output; fail → feedback iteration
graph.add_conditional_edges(
    "compliance_validate",
    compliance_router,          # Routing function
    {
        "pass": "privacy_unmask",
        "fail": "feedback_refine",
    }
)

graph.add_edge("privacy_unmask",     END)   # Output for human review

# Feedback iteration: return to narrative generation
graph.add_edge("feedback_refine",    "narrative_generate")

# ── Compile (enable checkpoint persistence) ──
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["privacy_unmask"],  # Human-in-the-loop interrupt point
)
```

### 3.3 Typology Detection Subgraph (TypologySubgraph)

Leverages LangGraph's **Send API** for dynamic parallelism: based on the Planning Agent's decisions, only the required Typology Agents are activated.

```python
from langgraph.constants import Send

def plan_to_typology_dispatch(state: SARState) -> list[Send]:
    """Dynamically dispatch to corresponding Typology Agents based on the execution plan"""
    sends = []
    for agent_name in state["active_typology_agents"]:
        sends.append(Send(agent_name, {
            "masked_data": state["masked_data"],
            "risk_indicators": state["risk_indicators"],
            "crime_types": state["crime_types"],
        }))
    return sends

typology_graph = StateGraph(TypologyState)
typology_graph.add_node("transaction_fraud",   transaction_fraud_agent)
typology_graph.add_node("payment_velocity",    payment_velocity_agent)
typology_graph.add_node("country_risk",        country_risk_agent)
typology_graph.add_node("text_content",        text_content_agent)
typology_graph.add_node("geo_anomaly",         geo_anomaly_agent)
typology_graph.add_node("account_health",      account_health_agent)
typology_graph.add_node("dispute_pattern",     dispute_pattern_agent)
typology_graph.add_node("typology_merge",      merge_typology_results)

# Dynamic parallel dispatch
typology_graph.add_conditional_edges(START, plan_to_typology_dispatch)

# All parallel Agents converge at the merge node
for agent in TYPOLOGY_AGENTS:
    typology_graph.add_edge(agent, "typology_merge")

typology_graph.add_edge("typology_merge", END)
```

### 3.4 Human-in-the-Loop Interrupts

LangGraph natively supports `interrupt_before` / `interrupt_after`, perfectly matching the paper's human-AI collaboration design:

```python
# Set interrupt points at compile time
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["privacy_unmask"],  # Pause before unmasking output, await human review
)

# Streamlit-side execution resume (after receiving human feedback)
# ── Narrative Review Page (pages/review.py) ──
def on_submit_feedback():
    """Investigator modifies the narrative in the Streamlit editor and clicks submit"""
    case_id = st.session_state["current_case_id"]
    feedback = st.session_state["investigator_feedback"]
    config = {"configurable": {"thread_id": case_id}}

    # Update state and resume graph execution
    app.update_state(config, {"human_feedback": feedback})
    with st.spinner("Regenerating narrative based on feedback..."):
        result = app.invoke(None, config)
    st.session_state["sar_result"] = result
    st.rerun()

# UI Components
st.text_area("Investigator Feedback", key="investigator_feedback")
st.button("Submit Feedback & Regenerate", on_click=on_submit_feedback)
```

---

## 4. Streamlit UI Design

### 4.1 UI Architecture Overview

Streamlit serves as the sole interface between investigators and Argus AI, handling all responsibilities: data input, workflow control, narrative review, feedback submission, and result visualization. It uses the **Streamlit Multi-Page App** pattern to organize pages.

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit App (app.py)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sidebar (Global Navigation)                           │  │
│  │  ┌──────────────────┐                                  │  │
│  │  │ 📄 Case Upload    │ ← JSON file upload / sample      │  │
│  │  │ 🔍 SAR Generate   │ ← One-click, real-time progress  │  │
│  │  │ ✏️ Narrative Review│ ← Draft + inline edit + feedback │  │
│  │  │ 📊 Dashboard      │ ← Crime type/risk/compliance viz │  │
│  │  │ 📋 History        │ ← SAR list + search + export     │  │
│  │  └──────────────────┘                                  │  │
│  │                                                        │  │
│  │  Settings Panel (Sidebar Bottom)                       │  │
│  │  • DeepSeek API Key Configuration                      │  │
│  │  • Compliance Score Threshold Adjustment               │  │
│  │  • Maximum Iteration Count Setting                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Main Content Area (dynamically rendered per page)     │  │
│  │                                                        │  │
│  │  st.session_state management:                          │  │
│  │  • current_case: dict     — Currently loaded case data │  │
│  │  • sar_result: dict       — LangGraph execution result │  │
│  │  • graph_status: str      — Graph execution status     │  │
│  │  • thread_id: str         — LangGraph thread ID        │  │
│  │  • iteration_count: int   — Current iteration round    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Page Design Details

#### 4.2.1 📄 Case Upload Page (`pages/1_Case_Upload.py`)

| Feature | Implementation |
|---|---|
| **JSON File Upload** | `st.file_uploader(type=["json"])` to accept case files uploaded by investigators |
| **Sample Data Selection** | `st.selectbox` to load preset samples from the `data/samples/` directory |
| **Data Preview** | `st.json()` displays raw JSON; `st.dataframe()` shows structured views of transactions, accounts, etc. |
| **Data Validation** | Auto-validates JSON Schema upon upload; uses `st.error()` / `st.success()` for validation feedback |
| **Load to Session** | On successful validation, writes to `st.session_state["current_case"]` and auto-navigates to the generation page |

#### 4.2.2 🔍 SAR Generation Page (`pages/2_SAR_Generate.py`)

| Feature | Implementation |
|---|---|
| **One-Click Generate** | `st.button("🚀 Start SAR Generation")` triggers LangGraph graph execution |
| **Real-Time Progress** | Uses `st.status()` + `st.write_stream()` to stream each Agent's execution status |
| **Agent Progress Tracking** | Custom `progress_tracker` component displaying a step bar: Data Ingestion ✅ → Privacy Masking ✅ → Crime Detection 🔄 → ... |
| **Intermediate Result Preview** | Collapsible `st.expander()` showing each Agent's output summary |
| **Error Handling** | Catches exceptions with `st.error()` display and retry support |

```python
# Core execution logic (pages/2_SAR_Generate.py)
if st.button("🚀 Start SAR Generation"):
    case_data = st.session_state["current_case"]
    thread_id = case_data["case_id"]
    config = {"configurable": {"thread_id": thread_id}}

    with st.status("Generating SAR narrative...", expanded=True) as status:
        # stream_mode="updates" retrieves results node by node
        for event in app.stream(
            {"raw_data": case_data, "case_id": thread_id},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                st.write(f"✅ **{node_name}** completed")
                with st.expander(f"{node_name} details", expanded=False):
                    st.json(node_output)

        status.update(label="SAR generation complete!", state="complete")

    # Save results to session
    st.session_state["sar_result"] = app.get_state(config).values
    st.session_state["thread_id"] = thread_id
```

#### 4.2.3 ✏️ Narrative Review Page (`pages/3_Narrative_Review.py`)

This is the **core page for Human-in-the-Loop**. The paper emphasizes that investigators must be able to review and modify AI-generated drafts:

| Feature | Implementation |
|---|---|
| **Narrative Display** | `st.markdown()` renders the formatted SAR narrative (Intro + Body + Conclusion) |
| **Inline Editing** | `st.text_area()` provides an editable text area for the narrative draft |
| **CoT Reasoning Chain** | `st.expander("🧠 Reasoning Process")` displays Chain-of-Thought for explainability |
| **Compliance Score** | `st.metric()` + `st.progress()` shows compliance validation score and pass/fail status |
| **Compliance Details** | `st.expander()` displays compliance check results across all dimensions |
| **Feedback Submission** | `st.text_area()` + `st.button()` to submit revision comments, triggering iterative regeneration |
| **Approve/Reject** | `st.button("✅ Approve")` / `st.button("❌ Reject")` for final decision |
| **Iteration Counter** | `st.info()` displays current iteration round / maximum rounds |

#### 4.2.4 📊 Analysis Dashboard (`pages/4_Analysis_Dashboard.py`)

| Feature | Implementation |
|---|---|
| **Crime Type Confidence** | `plotly` horizontal bar chart showing detection confidence for each crime type |
| **Transaction Timeline** | `plotly` timeline scatter plot annotating suspicious transaction nodes |
| **Risk Indicator Heatmap** | `plotly` heatmap displaying risk ratings across dimensions |
| **Related Entity Network** | `plotly` / `st.graphviz_chart` showing subject-account-entity relationship graph |
| **Typology Agent Results** | `st.columns()` multi-column layout, each column showing a Typology Agent's analysis summary |

#### 4.2.5 📋 History Page (`pages/5_History.py`)

| Feature | Implementation |
|---|---|
| **SAR List** | `st.dataframe()` displays historically generated SARs (ID, date, status, crime types, score) |
| **Search & Filter** | `st.text_input()` + `st.multiselect()` to filter by keywords, crime types, status |
| **Detail View** | Click a row to navigate to the complete SAR detail view |
| **JSON Export** | `st.download_button()` exports SAR results as JSON files |

### 4.3 Session State Management

Streamlit's `st.session_state` serves as the UI layer's state hub, bridging user interactions and LangGraph execution:

```python
# ui/session.py — Session State initialization and management

def init_session_state():
    """Called in app.py to initialize all session variables"""
    defaults = {
        "current_case": None,           # Current case JSON data
        "sar_result": None,             # LangGraph execution result
        "thread_id": None,              # LangGraph checkpoint thread_id
        "graph_status": "idle",         # idle / running / interrupted / completed
        "iteration_count": 0,           # Feedback iteration counter
        "history": [],                  # Historical SAR records list
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_case():
    """Reset current case-related state"""
    st.session_state["current_case"] = None
    st.session_state["sar_result"] = None
    st.session_state["thread_id"] = None
    st.session_state["graph_status"] = "idle"
    st.session_state["iteration_count"] = 0
```

### 4.4 Launch Method

```bash
# Launch the Streamlit application
streamlit run src/app.py

# app.py entry file structure
# ── src/app.py ──
import streamlit as st
from ui.session import init_session_state

st.set_page_config(
    page_title="Argus AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

st.title("🔍 Argus AI")
st.markdown("**AML Compliance Narrative Intelligence Platform** — Multi-Agent Collaborative SAR Auto-Generation System")

# Home page: System overview / Quick access
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SARs Pending Review", "3")
with col2:
    st.metric("Generated Today", "12")
with col3:
    st.metric("Avg Compliance Score", "0.87")
```

---

## 5. Agent Design Details

### 5.1 Unified Agent Pattern

Each Agent is represented as a **node function** in LangGraph, following a unified pattern:

```python
def agent_function(state: SARState) -> dict:
    """
    1. Read required inputs from state
    2. Execute the Agent's core logic (LLM call / ML inference / tool invocation)
    3. Return state fields to update (dict)
    """
    # ... core logic
    return {"field_to_update": new_value}
```

### 5.2 Agent Responsibilities and Implementation Strategies

#### 5.2.1 Data Ingestion Agent (`ingest`)

| Item | Description |
|---|---|
| **Input** | `raw_data` (JSON) |
| **Output** | `structured_data` |
| **Implementation** | Pure Python data transformation, no LLM dependency. Parses transaction records, account metadata, KYC information, and risk signals from JSON, outputting a standardized structure |

#### 5.2.2 AI-Privacy Guard Agent (`privacy_mask` / `privacy_unmask`)

| Item | Description |
|---|---|
| **Input** | `structured_data` |
| **Output** | `masked_data`, `mask_mapping` |
| **Implementation** | MVP stage uses Microsoft Presidio / spaCy NER to identify PII (names, SSN, addresses, account numbers, etc.) and generates an anonymization mapping table. To be later replaced with a self-trained RoBERTa+CRF model |
| **Bidirectional Operation** | `privacy_mask` masks sensitive info → LLM processes → `privacy_unmask` restores original data |

#### 5.2.3 Crime Type Detection Agent (`crime_detect`)

| Item | Description |
|---|---|
| **Input** | `masked_data` |
| **Output** | `risk_indicators`, `crime_types` |
| **Implementation** | Dual-component: ① Rule engine extracts risk indicators (abnormal transaction patterns, high-risk countries, unusual frequencies, etc.) ② scikit-learn ensemble models (RF / GBM) output crime type probability rankings |
| **LLM Assistance** | Optional: For emerging types not covered by rules, invoke DeepSeek for auxiliary classification |

#### 5.2.4 Planning Agent (`plan`)

| Item | Description |
|---|---|
| **Input** | `crime_types`, `risk_indicators`, `masked_data` |
| **Output** | `execution_plan`, `active_typology_agents` |
| **Implementation** | Invokes DeepSeek to decide, based on detected crime types and confidence levels: ① Which Typology Agents to activate ② Whether external intelligence is needed ③ Narrative focus and structural planning |

#### 5.2.5 Specialized Typology Detection Agents (7 Agents)

| Agent | Core Logic |
|---|---|
| **transaction_fraud** | Analyzes transaction amount/frequency/counterparty patterns; detects structuring, layering, and anomalous large transactions |
| **payment_velocity** | Computes transaction frequency/volume within time windows; detects sudden high-frequency activity |
| **country_risk** | Cross-references countries/regions involved in transactions against sanctions lists / FATF high-risk lists |
| **text_content** | NLP analysis of customer communications and transaction notes; detects suspicious keywords/semantic patterns |
| **geo_anomaly** | Detects geographic inconsistencies (login location vs. transaction location vs. registration location) |
| **account_health** | Evaluates account historical behavior baselines; detects anomalous deviations |
| **dispute_pattern** | Analyzes dispute/chargeback patterns; detects fraudulent disputes |

Each Agent uses a mix of **rule engines + ML models + DeepSeek reasoning**, outputting structured risk assessment reports.

#### 5.2.6 External Intelligence Agent (`external_intel`)

| Item | Description |
|---|---|
| **Input** | `crime_types`, `masked_data`, `execution_plan` |
| **Output** | `external_intel` |
| **Implementation** | Connects to external MCP Servers via MCP Client SDK, dynamically discovering and invoking data sources (adverse media, sanctions lists, regulatory bulletins). MVP stage simulates MCP calls with local JSON data |

#### 5.2.7 Narrative Generation Agent (`narrative_generate`)

| Item | Description |
|---|---|
| **Input** | `masked_data`, `typology_results`, `external_intel`, `execution_plan`, `human_feedback` (if any) |
| **Output** | `narrative_draft`, `narrative_intro`, `chain_of_thought` |
| **Implementation** | Invokes DeepSeek using Chain-of-Thought prompting to generate a FinCEN-compliant SAR narrative draft. Prompt templates include: narrative structure guidelines (5W1H), crime type context, regulatory requirements, and historical narrative references |

#### 5.2.8 Compliance Validation Agent (`compliance_validate`) — Agent-as-a-Judge

| Item | Description |
|---|---|
| **Input** | `narrative_draft`, `typology_results`, `masked_data` |
| **Output** | `compliance_result`, `compliance_score` |
| **Implementation** | Dual validation: ① **Rule-based validation** — Checks whether all required elements are present (subject info, date range, transaction amounts, crime types) ② **Semantic validation** — Invokes DeepSeek to evaluate narrative coherence, logical completeness, and regulatory compliance, producing structured scores |
| **Routing Logic** | score ≥ threshold → PASS → proceed to unmasking output; score < threshold → FAIL → generate improvement suggestions → Feedback Agent |

#### 5.2.9 Feedback Agent (`feedback_refine`)

| Item | Description |
|---|---|
| **Input** | `compliance_result`, `narrative_draft`, `human_feedback` |
| **Output** | Revision instructions for updating `narrative_draft`, `iteration_count` +1 |
| **Implementation** | Synthesizes compliance validation failure reasons + human feedback to generate specific narrative revision instructions, passed to the next round of narrative generation |

---

## 6. Dynamic Memory System

### 6.1 Three-Layer Memory Architecture

```
┌─────────────────────────────────────────────┐
│              MemoryManager                   │
│  (Unified interface; Agents are unaware of   │
│   underlying storage differences)            │
├─────────────┬──────────────┬────────────────┤
│ Regulatory  │ Historical   │ Typology       │
│ Memory      │ Narrative    │ Specific       │
│             │ Memory       │ Memory         │
├─────────────┼──────────────┼────────────────┤
│ ChromaDB    │ ChromaDB     │ SQLite         │
│ (Vector)    │ (Vector)     │ (Structured)   │
│             │              │                │
│ • AML Regs  │ • Hist. SARs │ • Risk Indicator│
│ • FinCEN    │ • Approval   │   Patterns     │
│   Guidelines│   Records    │ • Crime Type   │
│ • FATF      │ • Narrative  │   Features     │
│   Recs      │   Templates  │ • Detection    │
│             │              │   Thresholds   │
│             │              │ • Hist. Results│
└─────────────┴──────────────┴────────────────┘
```

### 6.2 Integration with LangGraph

The memory system is integrated as **Tools for LangGraph nodes**, with Agents accessing it through standard tool calls:

```python
@tool
def search_regulatory_memory(query: str) -> list[Document]:
    """Search the regulatory memory store"""

@tool
def search_historical_narratives(query: str, crime_type: str) -> list[Document]:
    """Search historical SAR narratives"""

@tool
def get_typology_patterns(crime_type: str) -> dict:
    """Retrieve historical analysis patterns for a specific crime type"""
```

---

## 7. LLM Gateway Design

### 7.1 DeepSeek-First Multi-Model Strategy

```python
class LLMGateway:
    """Unified LLM invocation entry point, supporting per-Agent-role routing to different models"""

    MODEL_ROUTING = {
        # Agent Role           → Model Configuration
        "planning":          {"provider": "deepseek", "model": "deepseek-chat"},
        "narrative":         {"provider": "deepseek", "model": "deepseek-chat"},
        "compliance_judge":  {"provider": "deepseek", "model": "deepseek-chat"},
        "crime_detection":   {"provider": "deepseek", "model": "deepseek-chat"},
        "typology":          {"provider": "deepseek", "model": "deepseek-chat"},
        "evaluation":        {"provider": "deepseek", "model": "deepseek-chat"},
    }
```

### 7.2 DeepSeek Integration

Connects to the DeepSeek API via LangChain's `ChatOpenAI` compatible interface:

```python
from langchain_openai import ChatOpenAI

deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
    temperature=0.1,        # Low randomness for SAR generation
    max_tokens=8192,
)
```

---

## 8. Data Models and Sample Data

### 8.1 Input Data Format (JSON)

```json
{
  "case_id": "CASE-2026-00142",
  "alert_date": "2026-01-15",
  "priority": "high",
  "subject": {
    "name": "John Michael Smith",
    "dob": "1985-03-22",
    "ssn": "123-45-6789",
    "address": "456 Oak Avenue, Miami, FL 33101",
    "phone": "+1-305-555-0142",
    "email": "jmsmith85@email.com",
    "occupation": "Self-employed consultant",
    "risk_rating": "high",
    "customer_since": "2023-06-10"
  },
  "accounts": [
    {
      "account_id": "ACC-9821034",
      "account_type": "checking",
      "opened_date": "2023-06-10",
      "balance": 45230.00,
      "currency": "USD",
      "branch": "Miami Downtown"
    },
    {
      "account_id": "ACC-9821035",
      "account_type": "savings",
      "opened_date": "2023-07-01",
      "balance": 128500.00,
      "currency": "USD",
      "branch": "Miami Downtown"
    }
  ],
  "transactions": [
    {
      "txn_id": "TXN-20260110-001",
      "date": "2026-01-10",
      "type": "wire_transfer_in",
      "amount": 9800.00,
      "currency": "USD",
      "from_account": "EXT-OFFSHORE-8831",
      "to_account": "ACC-9821034",
      "from_entity": "Global Trade Solutions Ltd",
      "from_country": "BZ",
      "description": "Consulting payment",
      "risk_flags": ["structured_amount", "high_risk_jurisdiction"]
    },
    {
      "txn_id": "TXN-20260111-002",
      "date": "2026-01-11",
      "type": "wire_transfer_in",
      "amount": 9700.00,
      "currency": "USD",
      "from_account": "EXT-OFFSHORE-8831",
      "to_account": "ACC-9821034",
      "from_entity": "Global Trade Solutions Ltd",
      "from_country": "BZ",
      "description": "Consulting payment Q4",
      "risk_flags": ["structured_amount", "high_risk_jurisdiction", "rapid_succession"]
    },
    {
      "txn_id": "TXN-20260112-003",
      "date": "2026-01-12",
      "type": "internal_transfer",
      "amount": 19000.00,
      "currency": "USD",
      "from_account": "ACC-9821034",
      "to_account": "ACC-9821035",
      "description": "Savings allocation",
      "risk_flags": ["layering_pattern"]
    },
    {
      "txn_id": "TXN-20260113-004",
      "date": "2026-01-13",
      "type": "wire_transfer_out",
      "amount": 15000.00,
      "currency": "USD",
      "from_account": "ACC-9821035",
      "to_account": "EXT-SHELL-7742",
      "to_entity": "Sunrise Holdings LLC",
      "to_country": "PA",
      "description": "Investment deposit",
      "risk_flags": ["shell_company_indicator", "high_risk_jurisdiction"]
    },
    {
      "txn_id": "TXN-20260113-005",
      "date": "2026-01-13",
      "type": "cash_withdrawal",
      "amount": 4500.00,
      "currency": "USD",
      "from_account": "ACC-9821034",
      "location": "ATM - Hialeah, FL",
      "risk_flags": ["geographic_anomaly"]
    }
  ],
  "kyc": {
    "verification_status": "verified",
    "last_review_date": "2025-06-10",
    "source_of_funds": "Consulting income",
    "expected_activity": "low_volume_domestic",
    "actual_activity_profile": "high_volume_international",
    "pep_status": false,
    "adverse_media_hits": [
      {
        "date": "2025-11-20",
        "source": "Financial Times",
        "summary": "Subject's former business partner indicted for wire fraud scheme"
      }
    ]
  },
  "communications": [
    {
      "date": "2026-01-09",
      "channel": "secure_message",
      "direction": "inbound",
      "content": "Need to move funds quickly before end of quarter. Can you expedite the international transfers?",
      "flagged": true,
      "flag_reason": "urgency_pressure"
    },
    {
      "date": "2026-01-14",
      "channel": "phone_note",
      "direction": "outbound",
      "content": "Customer called to inquire about increasing wire transfer limits. Became evasive when asked about purpose of recent transfers.",
      "flagged": true,
      "flag_reason": "evasive_behavior"
    }
  ],
  "alerts": [
    {
      "alert_id": "ALT-20260115-001",
      "type": "structuring",
      "severity": "high",
      "description": "Multiple incoming wire transfers just below $10,000 threshold within 48-hour window",
      "triggered_date": "2026-01-15"
    },
    {
      "alert_id": "ALT-20260115-002",
      "type": "high_risk_jurisdiction",
      "severity": "medium",
      "description": "Wire transfers originating from Belize and destined to Panama — both FATF-monitored jurisdictions",
      "triggered_date": "2026-01-15"
    }
  ],
  "related_entities": [
    {
      "entity_name": "Global Trade Solutions Ltd",
      "entity_type": "company",
      "jurisdiction": "Belize",
      "relationship": "funds_originator",
      "risk_notes": "Shell company characteristics — no verifiable business operations"
    },
    {
      "entity_name": "Sunrise Holdings LLC",
      "entity_type": "company",
      "jurisdiction": "Panama",
      "relationship": "funds_recipient",
      "risk_notes": "Registered 2025-09 — minimal operating history"
    }
  ]
}
```

### 8.2 SAR Output Format (JSON)

```json
{
  "sar_id": "SAR-2026-00142",
  "case_id": "CASE-2026-00142",
  "generated_at": "2026-01-16T10:30:00Z",
  "status": "review",
  "crime_types_detected": [
    {"type": "structuring", "confidence": 0.92},
    {"type": "money_laundering_layering", "confidence": 0.87},
    {"type": "shell_company_activity", "confidence": 0.78}
  ],
  "narrative": {
    "intro": "This SAR is being filed to report suspicious activity...",
    "body": "Between January 10, 2026, and January 13, 2026, the subject...",
    "conclusion": "Based on the above analysis, the described transaction patterns..."
  },
  "compliance_validation": {
    "score": 0.85,
    "status": "PASS",
    "checks": { ... }
  },
  "chain_of_thought": [ ... ],
  "metadata": {
    "iteration_count": 2,
    "agents_activated": [ ... ],
    "processing_time_seconds": 45
  }
}
```

---

## 9. Project Directory Structure

```
Argus-v2/
├── docs/
│   └── ARCHITECTURE.md              # This document
│
├── src/
│   ├── __init__.py
│   ├── app.py                       # Streamlit application entry (home page)
│   ├── config.py                    # Pydantic Settings configuration
│   │
│   ├── core/                        # Core abstractions
│   │   ├── __init__.py
│   │   ├── state.py                 # SARState TypedDict definition
│   │   ├── models.py                # Common data models (Pydantic)
│   │   └── llm_gateway.py           # Unified LLM Gateway (DeepSeek-first)
│   │
│   ├── graph/                       # LangGraph graph definitions
│   │   ├── __init__.py
│   │   ├── sar_graph.py             # Main graph: SARGenerationGraph
│   │   ├── typology_subgraph.py     # Subgraph: TypologySubgraph
│   │   └── routing.py               # Conditional routing functions
│   │
│   ├── agents/                      # Agent implementations (LangGraph nodes)
│   │   ├── __init__.py
│   │   ├── ingestion.py             # Data Ingestion Agent
│   │   ├── privacy_guard.py         # AI-Privacy Guard (mask/unmask)
│   │   ├── crime_detection.py       # Crime Type Detection Agent
│   │   ├── planning.py              # Planning Agent (orchestrator)
│   │   ├── narrative.py             # Narrative Generation Agent
│   │   ├── compliance.py            # Compliance Validation Agent (Agent-as-a-Judge)
│   │   ├── feedback.py              # Feedback Agent
│   │   ├── external_intel.py        # External Intelligence Agent (MCP)
│   │   └── typology/                # 7 Specialized Typology Detection Agents
│   │       ├── __init__.py
│   │       ├── transaction_fraud.py
│   │       ├── payment_velocity.py
│   │       ├── country_risk.py
│   │       ├── text_content.py
│   │       ├── geo_anomaly.py
│   │       ├── account_health.py
│   │       └── dispute_pattern.py
│   │
│   ├── infrastructure/              # Infrastructure
│   │   ├── __init__.py
│   │   ├── memory/                  # Three-layer dynamic memory
│   │   │   ├── __init__.py
│   │   │   ├── manager.py           # MemoryManager unified interface
│   │   │   ├── regulatory.py        # Regulatory memory (ChromaDB)
│   │   │   ├── historical.py        # Historical narrative memory (ChromaDB)
│   │   │   └── typology.py          # Typology-specific memory (SQLite)
│   │   ├── tools/                   # Analytical tools
│   │   │   ├── __init__.py
│   │   │   ├── risk_indicators.py   # Risk indicator extraction
│   │   │   ├── account_linking.py   # Account linking analysis
│   │   │   └── external_search.py   # External intelligence search
│   │   └── mcp_client.py            # MCP client
│   │
│   └── ui/                          # Streamlit UI layer
│       ├── __init__.py
│       ├── pages/                   # Streamlit multi-page
│       │   ├── 1_Case_Upload.py     # Case data upload and preview
│       │   ├── 2_SAR_Generate.py    # SAR generation workflow and real-time progress
│       │   ├── 3_Narrative_Review.py# Narrative draft review and feedback
│       │   ├── 4_Analysis_Dashboard.py # Risk analysis visualization
│       │   └── 5_History.py         # Historical SAR management
│       ├── components/              # Reusable UI components
│       │   ├── __init__.py
│       │   ├── case_viewer.py       # Case data display component
│       │   ├── narrative_editor.py  # Narrative editor component
│       │   ├── progress_tracker.py  # Agent execution progress component
│       │   └── risk_charts.py       # Risk chart component
│       └── session.py               # Session State management
│
├── data/
│   └── samples/                     # Sample data
│       ├── case_structuring.json    # Sample: Structuring transactions
│       ├── case_elder_exploit.json  # Sample: Elder financial exploitation
│       └── case_shell_company.json  # Sample: Shell company money laundering
│
├── prompts/                         # Prompt templates
│   ├── planning.yaml
│   ├── narrative_generation.yaml
│   ├── compliance_validation.yaml
│   └── crime_detection.yaml
│
├── evaluation/                      # Evaluation framework
│   ├── golden_datasets/             # Golden datasets
│   ├── scoring.py                   # Scoring logic
│   └── runner.py                    # Evaluation runner
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── pyproject.toml                   # Project dependencies and metadata
└── README.md
```

---

## 10. Key Dependencies

```toml
[project]
name = "Argus-v2"
requires-python = ">=3.11"

dependencies = [
    # ── LangGraph / LangChain ──
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",       # DeepSeek via OpenAI-compatible interface
    "langchain-community>=0.3.0",

    # ── UI ──
    "streamlit>=1.40.0",
    "plotly>=5.24.0",              # Analysis dashboard charts

    # ── Data / ML ──
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "scikit-learn>=1.5.0",

    # ── Memory / Vector ──
    "chromadb>=0.5.0",

    # ── Privacy (MVP) ──
    "presidio-analyzer>=2.2",
    "presidio-anonymizer>=2.2",
    "spacy>=3.7",

    # ── MCP ──
    "mcp>=1.0",

    # ── Utilities ──
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.6",
]
```

---

## 11. Implementation Roadmap

| Phase | Objective | Modules Involved | Expected Output |
|---|---|---|---|
| **Phase 0** | Project Skeleton | Project structure, configuration, State definition, LLM Gateway | Runnable empty shell project |
| **Phase 1** | Minimal Pipeline | Data Ingestion → Crime Type Detection (simplified) → Narrative Generation → Output. Three-node LangGraph linear graph | End-to-end runnable, generating initial SAR draft |
| **Phase 2** | Full Agent Suite | Planning Agent, 7 Typology Agents (subgraph), Compliance Validation Agent, Feedback Loop | Complete LangGraph main graph + subgraph |
| **Phase 3** | Security & Memory | AI-Privacy Guard, Three-layer Dynamic Memory, External Intelligence Agent (MCP) | Privacy compliance + context augmentation |
| **Phase 4** | Human-AI Collaboration | Human-in-the-Loop interrupts, Streamlit review pages, feedback interaction components | Interactive review workflow |
| **Phase 5** | Evaluation & Optimization | Offline evaluation framework, Agent-as-a-Judge online validation, prompt optimization | Quantifiable quality assurance |
