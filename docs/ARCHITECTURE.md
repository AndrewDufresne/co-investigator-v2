# Co-Investigator V2 — 系统架构设计文档

> 基于论文 *"Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* (arXiv:2509.08380v2) 的方法论实现。

---

## 1. 架构总览

### 1.1 设计原则

| 原则 | 说明 |
|---|---|
| **Agent 化分解** | 将 SAR 生成拆解为多个专业 Agent，每个 Agent 职责单一、可独立演进 |
| **LangGraph 驱动** | 所有 Agent 编排基于 LangGraph 状态图，实现可控、可观测的工作流 |
| **Human-in-the-Loop** | 人类调查人员始终在回路中，AI 生成草稿供审查，不做自动提交 |
| **隐私优先** | 敏感数据在进入 LLM 前必须经过 AI-Privacy Guard 匿名化 |
| **单体部署** | 单进程 Streamlit 应用，内部通过 LangGraph 编排，降低运维复杂度 |

### 1.2 技术选型

| 层面 | 选型 | 理由 |
|---|---|---|
| **语言** | Python 3.11+ | Agent / LLM / NLP 生态最成熟 |
| **Agent 编排** | LangGraph (v0.2+) | 提供有状态图执行、条件路由、人机交互断点、检查点持久化 |
| **LLM** | DeepSeek（主力） | 优先支持；通过统一 Gateway 可扩展其他模型 |
| **LLM 接入** | LangChain ChatModel 抽象 | LangGraph 原生支持，统一 DeepSeek / OpenAI / Anthropic 接口 |
| **隐私层** | RoBERTa + CRF（论文方案） | MVP 阶段先用 presidio / spaCy NER，后替换为自训练模型 |
| **犯罪类型检测** | scikit-learn (RF / GBM) | 论文指定树模型集成方法 |
| **记忆层** | ChromaDB (向量) + SQLite (结构化) | 轻量级，适合单体部署 |
| **UI 框架** | Streamlit | 快速构建数据应用 UI，原生支持交互组件、实时流式输出、Session State 状态管理 |
| **数据格式** | JSON | 输入 / 输出 / 样例数据统一 JSON |
| **配置管理** | Pydantic Settings + YAML | 类型安全的配置加载 |

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Application                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   UI Layer (Streamlit)                     │  │
│  │                                                           │  │
│  │  📄 案例上传页    — JSON 文件上传 / 样例选择               │  │
│  │  🔍 SAR 生成页    — 一键触发，实时流式展示 Agent 进度      │  │
│  │  ✏️ 叙述审查页    — 草稿展示 + 内联编辑 + 反馈提交         │  │
│  │  📊 分析仪表盘    — 犯罪类型、风险指标、合规评分可视化     │  │
│  │  📋 历史记录页    — 历史 SAR 列表 + 搜索 + 导出            │  │
│  │                                                           │  │
│  │  状态管理: st.session_state (案例数据/图执行状态/反馈)     │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              LangGraph Orchestration Layer                 │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              SARGenerationGraph (主图)               │  │  │
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
│  │  │  │                  └────────┘ (迭代)                │  │  │
│  │  └──┴──────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │         TypologySubgraph (子图，动态并行)           │  │  │
│  │  │                                                     │  │  │
│  │  │  [transaction_fraud]     [payment_velocity]         │  │  │
│  │  │  [country_risk]          [text_content]             │  │  │
│  │  │  [geo_anomaly]           [account_health]           │  │  │
│  │  │  [dispute_pattern]                                  │  │  │
│  │  │          ──── 全部汇聚 → [typology_merge] ────      │  │  │
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

## 3. LangGraph 核心设计

### 3.1 全局状态 (State Schema)

LangGraph 以 **TypedDict** 定义贯穿整个工作流的共享状态。所有 Agent（节点）读写同一个 State 对象：

```python
class SARState(TypedDict):
    # ── 输入数据 ──
    case_id: str                          # 案例唯一标识
    raw_data: dict                        # 原始 JSON 案例数据
    structured_data: dict                 # 结构化后的数据
    masked_data: dict                     # 匿名化后的数据
    mask_mapping: dict                    # 匿名化映射表 (用于脱敏还原)

    # ── 犯罪类型检测 ──
    risk_indicators: list[dict]           # 提取的风险指标
    crime_types: list[CrimeTypeResult]    # 检测到的犯罪类型 + 置信度

    # ── 规划 ──
    execution_plan: ExecutionPlan         # Planning Agent 生成的执行计划
    active_typology_agents: list[str]     # 需要激活的类型检测 Agent 列表

    # ── 类型检测结果 ──
    typology_results: dict[str, dict]     # 各类型 Agent 的分析结果

    # ── 外部情报 ──
    external_intel: list[dict]            # MCP 获取的外部情报

    # ── 叙述生成 ──
    narrative_draft: str                  # 当前叙述草稿
    narrative_intro: str                  # 叙述引言部分
    chain_of_thought: list[str]           # CoT 推理链

    # ── 合规验证 ──
    compliance_result: ComplianceResult   # 验证结果 (PASS/FAIL + 详情)
    compliance_score: float               # 合规评分

    # ── 人机交互 ──
    human_feedback: str | None            # 人类反馈内容
    iteration_count: int                  # 当前迭代轮次
    max_iterations: int                   # 最大迭代次数

    # ── 最终输出 ──
    final_narrative: str                  # 最终 SAR 叙述
    status: Literal["processing", "review", "approved", "rejected"]
    messages: Annotated[list, add_messages]  # Agent 间消息日志
```

### 3.2 主图 (SARGenerationGraph)

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(SARState)

# ── 注册节点 (每个节点对应一个 Agent 函数) ──
graph.add_node("ingest",              data_ingestion_agent)
graph.add_node("privacy_mask",        privacy_mask_agent)
graph.add_node("crime_detect",        crime_detection_agent)
graph.add_node("plan",                planning_agent)
graph.add_node("typology_analysis",   typology_subgraph)      # 子图
graph.add_node("external_intel",      external_intel_agent)
graph.add_node("narrative_generate",  narrative_generation_agent)
graph.add_node("compliance_validate", compliance_validation_agent)
graph.add_node("privacy_unmask",      privacy_unmask_agent)
graph.add_node("feedback_refine",     feedback_agent)

# ── 定义边 (线性 + 条件路由) ──
graph.add_edge(START,                 "ingest")
graph.add_edge("ingest",             "privacy_mask")
graph.add_edge("privacy_mask",       "crime_detect")
graph.add_edge("crime_detect",       "plan")
graph.add_edge("plan",               "typology_analysis")
graph.add_edge("typology_analysis",  "external_intel")
graph.add_edge("external_intel",     "narrative_generate")
graph.add_edge("narrative_generate", "compliance_validate")

# 条件路由: 合规验证通过 → 脱敏输出; 失败 → 反馈迭代
graph.add_conditional_edges(
    "compliance_validate",
    compliance_router,          # 路由函数
    {
        "pass": "privacy_unmask",
        "fail": "feedback_refine",
    }
)

graph.add_edge("privacy_unmask",     END)   # 输出供人类审查

# 反馈迭代: 回到叙述生成
graph.add_edge("feedback_refine",    "narrative_generate")

# ── 编译 (启用检查点持久化) ──
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["privacy_unmask"],  # 人机交互断点
)
```

### 3.3 类型检测子图 (TypologySubgraph)

利用 LangGraph 的 **Send API** 实现动态并行：根据 Planning Agent 的决策，仅激活需要的 Typology Agent。

```python
from langgraph.constants import Send

def plan_to_typology_dispatch(state: SARState) -> list[Send]:
    """根据执行计划动态分发到对应的 Typology Agent"""
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

# 动态并行分发
typology_graph.add_conditional_edges(START, plan_to_typology_dispatch)

# 所有并行 Agent 汇聚到 merge 节点
for agent in TYPOLOGY_AGENTS:
    typology_graph.add_edge(agent, "typology_merge")

typology_graph.add_edge("typology_merge", END)
```

### 3.4 Human-in-the-Loop 断点

LangGraph 原生支持 `interrupt_before` / `interrupt_after`，完美匹配论文的人机协作设计：

```python
# 编译时设置断点
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["privacy_unmask"],  # 在脱敏输出前暂停，等待人类审查
)

# Streamlit 端恢复执行 (接收人类反馈后)
# ── 叙述审查页 (pages/review.py) ──
def on_submit_feedback():
    """调查人员在 Streamlit 编辑器中修改叙述后点击提交"""
    case_id = st.session_state["current_case_id"]
    feedback = st.session_state["investigator_feedback"]
    config = {"configurable": {"thread_id": case_id}}

    # 更新状态并恢复图执行
    app.update_state(config, {"human_feedback": feedback})
    with st.spinner("正在根据反馈重新生成叙述..."):
        result = app.invoke(None, config)
    st.session_state["sar_result"] = result
    st.rerun()

# UI 组件
st.text_area("调查人员反馈", key="investigator_feedback")
st.button("提交反馈并重新生成", on_click=on_submit_feedback)
```

---

## 4. Streamlit UI 设计

### 4.1 UI 架构概览

Streamlit 作为调查人员与 Co-Investigator AI 交互的唯一界面，承担数据输入、流程控制、叙述审查、反馈提交和结果可视化的全部职责。采用 **Streamlit Multi-Page App** 模式组织页面。

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit App (app.py)                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Sidebar (全局导航)                                     │  │
│  │  ┌──────────────────┐                                  │  │
│  │  │ 📄 案例上传       │ ← JSON 文件上传 / 样例选择       │  │
│  │  │ 🔍 SAR 生成       │ ← 一键触发，实时流式进度         │  │
│  │  │ ✏️ 叙述审查       │ ← 草稿展示 + 内联编辑 + 反馈     │  │
│  │  │ 📊 分析仪表盘     │ ← 犯罪类型/风险/合规可视化       │  │
│  │  │ 📋 历史记录       │ ← 历史 SAR 列表 + 搜索 + 导出   │  │
│  │  └──────────────────┘                                  │  │
│  │                                                        │  │
│  │  Settings Panel (侧边栏底部)                            │  │
│  │  • DeepSeek API Key 配置                               │  │
│  │  • 合规评分阈值调节                                     │  │
│  │  • 最大迭代轮次设置                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Main Content Area (根据当前页面动态渲染)               │  │
│  │                                                        │  │
│  │  st.session_state 管理:                                │  │
│  │  • current_case: dict        — 当前加载的案例数据       │  │
│  │  • sar_result: dict          — LangGraph 执行结果       │  │
│  │  • graph_status: str         — 图执行状态               │  │
│  │  • thread_id: str            — LangGraph 线程 ID        │  │
│  │  • iteration_count: int      — 当前迭代轮次             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 页面详细设计

#### 4.2.1 📄 案例上传页 (`pages/1_📄_案例上传.py`)

| 功能 | 实现方式 |
|---|---|
| **JSON 文件上传** | `st.file_uploader(type=["json"])` 接收调查人员上传的案例文件 |
| **样例数据选择** | `st.selectbox` 从 `data/samples/` 目录加载预置样例 |
| **数据预览** | `st.json()` 展示原始 JSON，`st.dataframe()` 展示交易列表、账户信息等结构化视图 |
| **数据校验** | 上传后自动校验 JSON Schema，使用 `st.error()` / `st.success()` 反馈校验结果 |
| **加载到 Session** | 校验通过后写入 `st.session_state["current_case"]`，自动跳转到生成页 |

#### 4.2.2 🔍 SAR 生成页 (`pages/2_🔍_SAR生成.py`)

| 功能 | 实现方式 |
|---|---|
| **一键生成** | `st.button("🚀 开始生成 SAR")` 触发 LangGraph 图执行 |
| **实时进度** | 利用 `st.status()` + `st.write_stream()` 流式展示各 Agent 执行状态 |
| **Agent 进度追踪** | 自定义 `progress_tracker` 组件，以步骤条形式展示：数据摄取 ✅ → 隐私遮蔽 ✅ → 犯罪检测 🔄 → ... |
| **中间结果预览** | 通过 `st.expander()` 折叠展示每个 Agent 的输出摘要 |
| **错误处理** | 捕获异常后 `st.error()` 展示，支持重试 |

```python
# 核心执行逻辑示意 (pages/2_🔍_SAR生成.py)
if st.button("🚀 开始生成 SAR"):
    case_data = st.session_state["current_case"]
    thread_id = case_data["case_id"]
    config = {"configurable": {"thread_id": thread_id}}

    with st.status("正在生成 SAR 叙述...", expanded=True) as status:
        # stream_mode="updates" 可逐节点获取执行结果
        for event in app.stream(
            {"raw_data": case_data, "case_id": thread_id},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                st.write(f"✅ **{node_name}** 完成")
                with st.expander(f"{node_name} 详情", expanded=False):
                    st.json(node_output)

        status.update(label="SAR 生成完成!", state="complete")

    # 保存结果到 session
    st.session_state["sar_result"] = app.get_state(config).values
    st.session_state["thread_id"] = thread_id
```

#### 4.2.3 ✏️ 叙述审查页 (`pages/3_✏️_叙述审查.py`)

这是 **Human-in-the-Loop 的核心页面**，论文中强调调查人员必须能审查和修改 AI 生成的草稿：

| 功能 | 实现方式 |
|---|---|
| **叙述展示** | `st.markdown()` 渲染格式化的 SAR 叙述（Intro + Body + Conclusion） |
| **内联编辑** | `st.text_area()` 提供叙述草稿的可编辑文本区域 |
| **CoT 推理链** | `st.expander("🧠 推理过程")` 展示 Chain-of-Thought，增强可解释性 |
| **合规评分** | `st.metric()` + `st.progress()` 展示合规验证分数和通过/失败状态 |
| **合规详情** | `st.expander()` 展示各维度的合规检查结果 |
| **反馈提交** | `st.text_area()` + `st.button()` 提交修改意见，触发迭代重生成 |
| **批准/驳回** | `st.button("✅ 批准")` / `st.button("❌ 驳回")` 最终决策按钮 |
| **迭代计数** | `st.info()` 展示当前迭代轮次 / 最大轮次 |

#### 4.2.4 📊 分析仪表盘 (`pages/4_📊_分析仪表盘.py`)

| 功能 | 实现方式 |
|---|---|
| **犯罪类型置信度** | `plotly` 水平柱状图，展示各犯罪类型的检测置信度 |
| **交易时间线** | `plotly` 时间线散点图，标注可疑交易节点 |
| **风险指标热力图** | `plotly` 热力图，展示各维度的风险评级 |
| **关联实体网络** | `plotly` / `st.graphviz_chart` 展示主体-账户-实体关系图 |
| **Typology Agent 结果** | `st.columns()` 多列布局，每列展示一个 Typology Agent 的分析摘要 |

#### 4.2.5 📋 历史记录页 (`pages/5_📋_历史记录.py`)

| 功能 | 实现方式 |
|---|---|
| **SAR 列表** | `st.dataframe()` 展示历史生成的 SAR 列表（ID、日期、状态、犯罪类型、评分） |
| **搜索过滤** | `st.text_input()` + `st.multiselect()` 按关键词、犯罪类型、状态筛选 |
| **详情查看** | 点击行跳转到对应 SAR 的完整详情视图 |
| **JSON 导出** | `st.download_button()` 导出 SAR 结果为 JSON 文件 |

### 4.3 Session State 管理

Streamlit 的 `st.session_state` 作为 UI 层的状态中心，桥接用户交互与 LangGraph 执行：

```python
# ui/session.py — Session State 初始化与管理

def init_session_state():
    """在 app.py 中调用，初始化所有 session 变量"""
    defaults = {
        "current_case": None,           # 当前案例 JSON 数据
        "sar_result": None,             # LangGraph 执行结果
        "thread_id": None,              # LangGraph checkpoint thread_id
        "graph_status": "idle",         # idle / running / interrupted / completed
        "iteration_count": 0,           # 反馈迭代计数
        "history": [],                  # 历史 SAR 记录列表
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_case():
    """重置当前案例相关状态"""
    st.session_state["current_case"] = None
    st.session_state["sar_result"] = None
    st.session_state["thread_id"] = None
    st.session_state["graph_status"] = "idle"
    st.session_state["iteration_count"] = 0
```

### 4.4 启动方式

```bash
# 启动 Streamlit 应用
streamlit run src/app.py

# app.py 入口文件结构
# ── src/app.py ──
import streamlit as st
from ui.session import init_session_state

st.set_page_config(
    page_title="Co-Investigator AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

st.title("🔍 Co-Investigator AI")
st.markdown("**AML 合规叙述智能生成平台** — 基于多 Agent 协作的 SAR 自动化生成系统")

# 主页: 系统概览 / 快速入口
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("待审查 SAR", "3")
with col2:
    st.metric("今日生成", "12")
with col3:
    st.metric("平均合规评分", "0.87")
```

---

## 5. Agent 详细设计

### 5.1 Agent 统一模式

每个 Agent 在 LangGraph 中表现为一个 **节点函数**，遵循统一模式：

```python
def agent_function(state: SARState) -> dict:
    """
    1. 从 state 中读取所需输入
    2. 执行本 Agent 的核心逻辑 (LLM 调用 / ML 推理 / 工具调用)
    3. 返回需要更新的 state 字段 (dict)
    """
    # ... 核心逻辑
    return {"field_to_update": new_value}
```

### 5.2 各 Agent 职责与实现策略

#### 5.2.1 数据摄取 Agent (`ingest`)

| 项 | 说明 |
|---|---|
| **输入** | `raw_data` (JSON) |
| **输出** | `structured_data` |
| **实现** | 纯 Python 数据转换，不依赖 LLM。解析 JSON 中的交易记录、账户元数据、KYC 信息、风险信号，输出标准化结构 |

#### 5.2.2 AI-Privacy Guard Agent (`privacy_mask` / `privacy_unmask`)

| 项 | 说明 |
|---|---|
| **输入** | `structured_data` |
| **输出** | `masked_data`, `mask_mapping` |
| **实现** | MVP 阶段使用 Microsoft Presidio / spaCy NER 识别 PII（姓名、SSN、地址、账号等），生成匿名化映射表。后期替换为 RoBERTa+CRF 自训练模型 |
| **双向操作** | `privacy_mask` 遮蔽敏感信息 → LLM 处理 → `privacy_unmask` 还原真实数据 |

#### 5.2.3 犯罪类型检测 Agent (`crime_detect`)

| 项 | 说明 |
|---|---|
| **输入** | `masked_data` |
| **输出** | `risk_indicators`, `crime_types` |
| **实现** | 双组件：① 规则引擎提取风险指标（异常交易模式、高风险国家、异常频率等）② scikit-learn 集成模型（RF / GBM）输出犯罪类型概率排名 |
| **LLM 辅助** | 可选：对于规则无法覆盖的新兴类型，调用 DeepSeek 做辅助分类 |

#### 5.2.4 规划 Agent (`plan`)

| 项 | 说明 |
|---|---|
| **输入** | `crime_types`, `risk_indicators`, `masked_data` |
| **输出** | `execution_plan`, `active_typology_agents` |
| **实现** | 调用 DeepSeek，基于检测到的犯罪类型和置信度，决定：① 需要激活哪些 Typology Agent ② 是否需要外部情报 ③ 叙述重点和结构规划 |

#### 5.2.5 专业类型检测 Agent (7个)

| Agent | 核心逻辑 |
|---|---|
| **transaction_fraud** | 分析交易金额/频率/对手方模式，检测结构化交易、分层、异常大额 |
| **payment_velocity** | 统计时间窗口内交易频率/量级，检测突发性高频活动 |
| **country_risk** | 比对交易涉及的国家/地区与制裁名单 / FATF 高风险名单 |
| **text_content** | NLP 分析客户通信、交易备注，检测可疑关键词/语义模式 |
| **geo_anomaly** | 检测地理位置不一致（登录地 vs 交易地 vs 注册地） |
| **account_health** | 评估账户历史行为基线，检测异常偏离 |
| **dispute_pattern** | 分析争议/退单模式，检测欺诈性争议 |

每个 Agent 混合使用 **规则引擎 + ML 模型 + DeepSeek 推理**，输出结构化的风险评估报告。

#### 5.2.6 外部情报 Agent (`external_intel`)

| 项 | 说明 |
|---|---|
| **输入** | `crime_types`, `masked_data`, `execution_plan` |
| **输出** | `external_intel` |
| **实现** | 通过 MCP Client SDK 连接外部 MCP Server，动态发现和调用数据源（负面新闻、制裁名单、监管公告）。MVP 阶段模拟 MCP 调用，用本地 JSON 数据替代 |

#### 5.2.7 叙述生成 Agent (`narrative_generate`)

| 项 | 说明 |
|---|---|
| **输入** | `masked_data`, `typology_results`, `external_intel`, `execution_plan`, `human_feedback` (如有) |
| **输出** | `narrative_draft`, `narrative_intro`, `chain_of_thought` |
| **实现** | 调用 DeepSeek，使用 Chain-of-Thought 提示，生成符合 FinCEN 格式的 SAR 叙述草稿。提示模板包含：叙述结构指引 (5W1H)、犯罪类型上下文、监管要求、历史叙述参考 |

#### 5.2.8 合规验证 Agent (`compliance_validate`) — Agent-as-a-Judge

| 项 | 说明 |
|---|---|
| **输入** | `narrative_draft`, `typology_results`, `masked_data` |
| **输出** | `compliance_result`, `compliance_score` |
| **实现** | 双重验证：① **规则验证** — 检查必要元素（主体信息、日期范围、交易金额、犯罪类型）是否齐全 ② **语义验证** — 调用 DeepSeek 评估叙述连贯性、逻辑完整性、监管合规性，输出结构化评分 |
| **路由逻辑** | score ≥ 阈值 → PASS → 进入脱敏输出; score < 阈值 → FAIL → 生成改进建议 → 反馈Agent |

#### 5.2.9 反馈 Agent (`feedback_refine`)

| 项 | 说明 |
|---|---|
| **输入** | `compliance_result`, `narrative_draft`, `human_feedback` |
| **输出** | 更新 `narrative_draft` 的修改指令，`iteration_count` +1 |
| **实现** | 综合合规验证的失败原因 + 人类反馈，生成具体的叙述修订指令，传递给下一轮叙述生成 |

---

## 6. 动态记忆系统

### 6.1 三层记忆架构

```
┌─────────────────────────────────────────────┐
│              MemoryManager                   │
│  (统一接口，Agent 不感知底层存储差异)         │
├─────────────┬──────────────┬────────────────┤
│ Regulatory  │ Historical   │ Typology       │
│ Memory      │ Narrative    │ Specific       │
│             │ Memory       │ Memory         │
├─────────────┼──────────────┼────────────────┤
│ ChromaDB    │ ChromaDB     │ SQLite         │
│ (向量检索)   │ (向量检索)    │ (结构化查询)   │
│             │              │                │
│ • AML 法规  │ • 历史 SAR   │ • 风险指标模式  │
│ • FinCEN    │ • 审批记录   │ • 犯罪类型特征  │
│   指南      │ • 叙述模板   │ • 检测阈值     │
│ • FATF 建议 │              │ • 历史分析结果  │
└─────────────┴──────────────┴────────────────┘
```

### 6.2 与 LangGraph 集成

记忆系统作为 **LangGraph 节点的工具 (Tool)** 接入，Agent 通过标准工具调用访问：

```python
@tool
def search_regulatory_memory(query: str) -> list[Document]:
    """搜索监管法规记忆库"""

@tool
def search_historical_narratives(query: str, crime_type: str) -> list[Document]:
    """搜索历史 SAR 叙述"""

@tool
def get_typology_patterns(crime_type: str) -> dict:
    """获取特定犯罪类型的历史分析模式"""
```

---

## 7. LLM Gateway 设计

### 7.1 DeepSeek 优先的多模型策略

```python
class LLMGateway:
    """统一 LLM 调用入口，支持按 Agent 角色路由到不同模型"""

    MODEL_ROUTING = {
        # Agent 角色         → 模型配置
        "planning":          {"provider": "deepseek", "model": "deepseek-chat"},
        "narrative":         {"provider": "deepseek", "model": "deepseek-chat"},
        "compliance_judge":  {"provider": "deepseek", "model": "deepseek-chat"},
        "crime_detection":   {"provider": "deepseek", "model": "deepseek-chat"},
        "typology":          {"provider": "deepseek", "model": "deepseek-chat"},
        "evaluation":        {"provider": "deepseek", "model": "deepseek-chat"},
    }
```

### 7.2 DeepSeek 接入方式

通过 LangChain 的 `ChatOpenAI` 兼容接口接入 DeepSeek API：

```python
from langchain_openai import ChatOpenAI

deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
    temperature=0.1,        # SAR 生成需要低随机性
    max_tokens=8192,
)
```

---

## 8. 数据模型与样例数据

### 8.1 输入数据格式 (JSON)

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

### 8.2 SAR 输出格式 (JSON)

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

## 9. 项目目录结构

```
co-investigator-v2/
├── docs/
│   └── ARCHITECTURE.md              # 本文档
│
├── src/
│   ├── __init__.py
│   ├── app.py                       # Streamlit 应用入口 (主页)
│   ├── config.py                    # Pydantic Settings 配置
│   │
│   ├── core/                        # 核心抽象
│   │   ├── __init__.py
│   │   ├── state.py                 # SARState TypedDict 定义
│   │   ├── models.py                # 通用数据模型 (Pydantic)
│   │   └── llm_gateway.py           # LLM 统一网关 (DeepSeek 优先)
│   │
│   ├── graph/                       # LangGraph 图定义
│   │   ├── __init__.py
│   │   ├── sar_graph.py             # 主图: SARGenerationGraph
│   │   ├── typology_subgraph.py     # 子图: TypologySubgraph
│   │   └── routing.py               # 条件路由函数
│   │
│   ├── agents/                      # Agent 实现 (LangGraph 节点)
│   │   ├── __init__.py
│   │   ├── ingestion.py             # 数据摄取 Agent
│   │   ├── privacy_guard.py         # AI-Privacy Guard (mask/unmask)
│   │   ├── crime_detection.py       # 犯罪类型检测 Agent
│   │   ├── planning.py              # 规划 Agent (编排器)
│   │   ├── narrative.py             # 叙述生成 Agent
│   │   ├── compliance.py            # 合规验证 Agent (Agent-as-a-Judge)
│   │   ├── feedback.py              # 反馈 Agent
│   │   ├── external_intel.py        # 外部情报 Agent (MCP)
│   │   └── typology/                # 7 个专业类型检测 Agent
│   │       ├── __init__.py
│   │       ├── transaction_fraud.py
│   │       ├── payment_velocity.py
│   │       ├── country_risk.py
│   │       ├── text_content.py
│   │       ├── geo_anomaly.py
│   │       ├── account_health.py
│   │       └── dispute_pattern.py
│   │
│   ├── infrastructure/              # 基础设施
│   │   ├── __init__.py
│   │   ├── memory/                  # 三层动态记忆
│   │   │   ├── __init__.py
│   │   │   ├── manager.py           # MemoryManager 统一接口
│   │   │   ├── regulatory.py        # 监管记忆 (ChromaDB)
│   │   │   ├── historical.py        # 历史叙述记忆 (ChromaDB)
│   │   │   └── typology.py          # 类型特定记忆 (SQLite)
│   │   ├── tools/                   # 分析工具
│   │   │   ├── __init__.py
│   │   │   ├── risk_indicators.py   # 风险指标提取
│   │   │   ├── account_linking.py   # 账户关联分析
│   │   │   └── external_search.py   # 外部情报搜索
│   │   └── mcp_client.py            # MCP 客户端
│   │
│   └── ui/                          # Streamlit UI 层
│       ├── __init__.py
│       ├── pages/                   # Streamlit 多页面
│       │   ├── 1_📄_案例上传.py      # 案例数据上传与预览
│       │   ├── 2_🔍_SAR生成.py      # SAR 生成流程与实时进度
│       │   ├── 3_✏️_叙述审查.py     # 叙述草稿审查与反馈
│       │   ├── 4_📊_分析仪表盘.py   # 风险分析可视化
│       │   └── 5_📋_历史记录.py     # 历史 SAR 管理
│       ├── components/              # 可复用 UI 组件
│       │   ├── __init__.py
│       │   ├── case_viewer.py       # 案例数据展示组件
│       │   ├── narrative_editor.py  # 叙述编辑器组件
│       │   ├── progress_tracker.py  # Agent 执行进度组件
│       │   └── risk_charts.py       # 风险图表组件
│       └── session.py               # Session State 管理
│
├── data/
│   └── samples/                     # 样例数据
│       ├── case_structuring.json    # 样例: 结构化交易
│       ├── case_elder_exploit.json  # 样例: 老年人金融剥削
│       └── case_shell_company.json  # 样例: 壳公司洗钱
│
├── prompts/                         # 提示模板
│   ├── planning.yaml
│   ├── narrative_generation.yaml
│   ├── compliance_validation.yaml
│   └── crime_detection.yaml
│
├── evaluation/                      # 评估框架
│   ├── golden_datasets/             # 黄金数据集
│   ├── scoring.py                   # 评分逻辑
│   └── runner.py                    # 评估运行器
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── pyproject.toml                   # 项目依赖与元数据
└── README.md
```

---

## 10. 关键依赖

```toml
[project]
name = "co-investigator-v2"
requires-python = ">=3.11"

dependencies = [
    # ── LangGraph / LangChain ──
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",       # DeepSeek 通过 OpenAI 兼容接口
    "langchain-community>=0.3.0",

    # ── UI ──
    "streamlit>=1.40.0",
    "plotly>=5.24.0",              # 分析仪表盘图表

    # ── 数据 / ML ──
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "scikit-learn>=1.5.0",

    # ── 记忆 / 向量 ──
    "chromadb>=0.5.0",

    # ── 隐私 (MVP) ──
    "presidio-analyzer>=2.2",
    "presidio-anonymizer>=2.2",
    "spacy>=3.7",

    # ── MCP ──
    "mcp>=1.0",

    # ── 工具 ──
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

## 11. 实现路线图

| 阶段 | 目标 | 涉及模块 | 预期产出 |
|---|---|---|---|
| **Phase 0** | 项目骨架 | 项目结构、配置、State 定义、LLM Gateway | 可运行的空壳项目 |
| **Phase 1** | 最小流程 | 数据摄取 → 犯罪类型检测(简化) → 叙述生成 → 输出。三个节点的 LangGraph 线性图 | 端到端可跑通，生成初步 SAR 草稿 |
| **Phase 2** | Agent 完整化 | Planning Agent、7 个 Typology Agent (子图)、合规验证 Agent、反馈循环 | 完整 LangGraph 主图 + 子图 |
| **Phase 3** | 安全与记忆 | AI-Privacy Guard、三层动态记忆、外部情报 Agent (MCP) | 隐私合规 + 上下文增强 |
| **Phase 4** | 人机协作 | Human-in-the-Loop 断点、Streamlit 审查页面、反馈交互组件 | 可交互的审查流程 |
| **Phase 5** | 评估与优化 | 离线评估框架、Agent-as-a-Judge 在线验证、提示优化 | 可量化的质量保证 |
