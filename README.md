# Eval-Driven Agentic Code Reviewer

An autonomous, enterprise-grade agent that reviews source code and architectures, gathers evidence with independent tools, reasons through a stateful LangGraph orchestration pipeline, and continuously scores outputs against "Senior Engineer Taste" criteria using DeepEval.

Features a full-stack interactive UI powered by Streamlit and persistent thread-level state storage with SQLite.

---

## Architecture & Roadmap Status

- [x] **Phase 0** — Manual tool-calling loop baseline (`src/phase0_toolcall_agent.py`)
- [x] **Phase 1** — LangGraph stateful orchestration + self-correction loops (`src/phase1_langgraph_agent.py`)
- [x] **Phase 2** — Evaluation harness with DeepEval & senior taste metrics (`src/phase2_evaluation.py`)
- [x] **Phase 3** — Hybrid RAG retrieval (Dense Vector + BM25 Lexical + Cross-Encoder Reranker) (`src/phase3_hybrid_rag.py`)
- [x] **Phase 4** — Full Autonomous Multi-Turn Agent (`src/phase4_agent.py`)
- [x] **Phase 5** — Model Context Protocol (MCP) Client & Server Integration (`src/phase5_mcp_client.py`)
- [x] **Phase 6** — Interactive Streamlit Review Studio & Cloud Deployment (`app.py`)

---

## Key Capabilities

1. **Autonomous Code Analysis & Diff View**: Syntax verification, vulnerability detection, and unified Git-style diff suggestions.
2. **PyTest Suite Generator**: Automated unit test creation with edge case coverage.
3. **Stateful Multi-Session Memory**: Isolated conversations and checkpoints backed by local SQLite (`agent_memory.db`).
4. **Corporate-Ready UI**: Custom enterprise minimalist Streamlit interface without default headers, footers, or menu bars.

---

## Local Setup & Installation

This project is optimized for Google Gemini models. You can obtain an API key via [Google AI Studio](https://aistudio.google.com/apikey).

### 1. Clone & Set Up Virtual Environment

```bash
git clone [https://github.com/KULLANICI_ADIN/autonomous-code-studio.git](https://github.com/KULLANICI_ADIN/autonomous-code-studio.git)
cd autonomous-code-studio

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
2. Dependencies
Bash
pip install -r requirements.txt
3. Execution
Launch the interactive Streamlit Studio:

Bash
streamlit run app.py
Cloud Deployment (Streamlit Community Cloud)
Link this repository to share.streamlit.io.

Set the main file path to app.py.

In Advanced Settings -> Secrets, provide:

Ini, TOML
GEMINI_API_KEY = "your_actual_api_key_here"
Deploy the application.
"@ -Encoding UTF8