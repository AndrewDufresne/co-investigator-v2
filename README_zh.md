# 🔍 Argus V2

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **基于多 Agent 协作的 AML 可疑活动报告 (SAR) 叙述自动生成框架。**

基于论文：*"Argus AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* (arXiv:2509.08380v2)。

**🌐 [English README](README.md)**

---

## ✨ 核心特性

- **多 Agent 架构** — SAR 生成拆解为专业 Agent（数据摄取、犯罪检测、规划、类型分析、叙述生成、合规验证），通过 LangGraph 状态图编排
- **人机协作 (Human-in-the-Loop)** — 调查人员审查和编辑 AI 生成的草稿，反馈触发迭代优化
- **隐私优先设计** — AI-Privacy Guard 在 LLM 处理前匿名化敏感数据 (PII)，处理后还原
- **动态类型分析** — 7 个专业类型检测 Agent（交易欺诈、支付频率、国家风险、文本内容、地理异常、账户健康、争议模式），基于检测到的犯罪类型动态激活
- **合规验证** — Agent-as-a-Judge 模式，规则验证 + 语义验证双重保障，确保符合 FinCEN 规范
- **思维链可解释性** — 完整的 CoT 推理链，增强透明度和可审计性
- **三层动态记忆** — 监管记忆、历史叙述记忆、类型特定记忆，实现上下文增强生成

---

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit 应用                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 UI 层 (5 个页面)                        │  │
│  │  📄 案例上传 → 🔍 SAR 生成 → ✏️ 叙述审查               │  │
│  │  📊 分析仪表盘 → 📋 历史记录                            │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │            LangGraph 编排层                             │  │
│  │                                                       │  │
│  │  [摄取] → [隐私遮蔽] → [犯罪检测] → [规划]             │  │
│  │    → [类型子图] → [外部情报]                            │  │
│  │    → [叙述生成] → [合规验证]                            │  │
│  │    → 通过: [隐私还原] → 结束                            │  │
│  │    → 失败: [反馈优化] → (迭代)                          │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌───────────────────────▼───────────────────────────────┐  │
│  │                基础设施层                               │  │
│  │  隐私守卫 │ LLM 网关 │ 动态记忆 │ MCP 客户端            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

完整架构设计请参阅：
- 📖 [架构设计文档 (中文)](docs/Architecture_zh.md)
- 📖 [Architecture Document (English)](docs/Architecture_en.md)

---

## 🚀 快速开始

### 前置要求

- Python 3.11+
- [DeepSeek API Key](https://platform.deepseek.com/)

### 1. 克隆并安装

```bash
git clone https://github.com/your-org/argus-v2.git
cd argus-v2
pip install -e .
```

### 2. 配置环境

复制 `.env.example` 为 `.env` 并设置 API 密钥：

```bash
cp .env.example .env
```

```env
DEEPSEEK_API_KEY=your-key-here
```

**可用配置项：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 基础 URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `LLM_TEMPERATURE` | `0.1` | LLM 温度参数（SAR 生成需要低随机性） |
| `COMPLIANCE_SCORE_THRESHOLD` | `0.75` | 合规验证通过的最低分数 |
| `MAX_ITERATIONS` | `3` | 最大反馈迭代轮次 |

### 3. 运行应用

```bash
streamlit run src/app.py
```

应用将在 `http://localhost:8501` 启动。

---

## 📁 项目结构

```
Argus-v2/
├── src/
│   ├── app.py                  # Streamlit 应用入口
│   ├── config.py               # Pydantic Settings 配置
│   ├── core/                   # 核心抽象
│   │   ├── state.py            #   SARState TypedDict 定义
│   │   ├── models.py           #   Pydantic 数据模型
│   │   └── llm_gateway.py      #   统一 LLM 网关 (DeepSeek 优先)
│   ├── graph/                  # LangGraph 工作流定义
│   │   ├── sar_graph.py        #   主图: SARGenerationGraph
│   │   ├── typology_subgraph.py#   子图: TypologySubgraph
│   │   └── routing.py          #   条件路由函数
│   ├── agents/                 # Agent 实现 (LangGraph 节点)
│   │   ├── ingestion.py        #   数据摄取 Agent
│   │   ├── privacy_guard.py    #   AI-Privacy Guard (遮蔽/还原)
│   │   ├── crime_detection.py  #   犯罪类型检测 Agent
│   │   ├── planning.py         #   规划 Agent (编排器)
│   │   ├── narrative.py        #   叙述生成 Agent
│   │   ├── compliance.py       #   合规验证 Agent (Agent-as-a-Judge)
│   │   ├── feedback.py         #   反馈 Agent
│   │   ├── external_intel.py   #   外部情报 Agent (MCP)
│   │   └── typology/           #   7 个专业类型检测 Agent
│   ├── infrastructure/         # 基础设施层
│   │   └── memory/             #   三层动态记忆
│   └── ui/                     # Streamlit UI 层
│       ├── pages/              #   多页面应用 (5 个页面)
│       ├── components/         #   可复用 UI 组件
│       └── session.py          #   Session State 管理
├── data/samples/               # 样例案例数据 (JSON)
├── prompts/                    # 提示模板 (YAML)
├── evaluation/                 # 评估框架
├── tests/                      # 单元测试 & 集成测试
├── docs/                       # 架构文档
├── pyproject.toml              # 项目依赖与元数据
└── README.md
```

---

## 🧪 测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行所有测试
pytest

# 运行冒烟测试
pytest tests/smoke_test.py

# 仅运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/
```

---

## 🔧 技术栈

| 层面 | 技术 | 用途 |
|---|---|---|
| **Agent 编排** | LangGraph v0.2+ | 有状态图执行、条件路由、人机交互断点 |
| **LLM** | DeepSeek (通过 LangChain) | 叙述生成、规划、合规验证的主力 LLM |
| **UI** | Streamlit | 调查人员交互界面，支持实时流式输出 |
| **犯罪检测** | scikit-learn (RF/GBM) | 基于树模型的犯罪类型集成分类 |
| **隐私** | Presidio / spaCy NER | PII 检测与匿名化 (MVP 阶段) |
| **记忆** | ChromaDB + SQLite | 向量检索 + 结构化存储 |
| **外部情报** | MCP Client SDK | 动态外部数据源集成 |
| **可视化** | Plotly | 风险图表、交易时间线、热力图 |

---

## 📋 实现路线图

| 阶段 | 目标 | 状态 |
|---|---|---|
| **Phase 0** | 项目骨架 — 结构、配置、状态定义、LLM 网关 | ✅ 完成 |
| **Phase 1** | 最小流程 — 摄取 → 犯罪检测 → 叙述生成 → 输出 | ✅ 完成 |
| **Phase 2** | Agent 完整化 — 规划、7 个类型 Agent、合规验证、反馈循环 | ✅ 完成 |
| **Phase 3** | 安全与记忆 — 隐私守卫、三层记忆、外部情报 (MCP) | 🔄 进行中 |
| **Phase 4** | 人机协作 — HITL 断点、审查页面、反馈交互 UI | ⬚ 计划中 |
| **Phase 5** | 评估与优化 — 离线评估、Agent-as-a-Judge、提示优化 | ⬚ 计划中 |

---

## 📄 许可

本项目仅用于研究和教育目的。

---

## 📚 参考

- 论文：*"Argus AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives"* — [arXiv:2509.08380v2](https://arxiv.org/abs/2509.08380v2)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [DeepSeek API](https://platform.deepseek.com/)
