"""
Autonomous Code Analysis & Architectural Review Studio
Enterprise Edition with PyTest Generator & Diff View
"""

import os
import re
import asyncio
import streamlit as st

# Import verified asynchronous orchestration pipeline from main.py
from main import run_agent_query

# 1. Page Configuration (Strictly clean, no emojis)
st.set_page_config(
    page_title="Autonomous Code Review Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Enterprise Minimalist CSS
st.markdown("""
<style>
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 24px;
        font-weight: 400;
    }
    .stAppDeployButton {
        display: none !important;
    }
    
    #MainMenu {
        visibility: hidden !important;
    }
    
    footer {
        visibility: hidden !important;
    }
    
    header {
        visibility: hidden !important;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .phase-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        font-family: monospace;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .metric-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar: Configuration, Audit Gates & Snippets
with st.sidebar:
    st.markdown("### System Configuration")
    
    session_id = st.text_input(
        "Active Thread ID:",
        value="production_review_alpha",
        help="Persistence key in SQLite (agent_memory.db) across restarts."
    )
    
    st.markdown("---")
    st.markdown("### Specialized Audit Modules")
    enable_owasp = st.checkbox("OWASP Security & Vulnerability Gate", value=True)
    enable_pep8 = st.checkbox("PEP 8 & Type Hint Integrity Gate", value=True)
    
    st.markdown("---")
    st.markdown("### Code Snippet Templates")
    
    preset_choice = st.selectbox(
        "Select benchmark case:",
        [
            "Custom Code",
            "SQL Injection & Insecure Secret (OWASP)",
            "Unsafe HTTP Client",
            "Unclosed Resource (I/O)",
            "Unbounded Recursion"
        ]
    )
    
    sample_codes = {
        "SQL Injection & Insecure Secret (OWASP)": (
            "import sqlite3\n\n"
            "SECRET_TOKEN = 'sk-live-9948271827419'\n\n"
            "def query_user_account(db_conn, username):\n"
            "    cursor = db_conn.cursor()\n"
            "    sql = f'SELECT * FROM accounts WHERE user = \"{username}\"'\n"
            "    cursor.execute(sql)\n"
            "    return cursor.fetchall()"
        ),
        "Unsafe HTTP Client": (
            "import requests\n\n"
            "def fetch_user_data(user_id):\n"
            "    url = 'https://api.example.com/users/' + str(user_id)\n"
            "    response = requests.get(url)\n"
            "    if response.status_code == 200:\n"
            "        return response.json()\n"
            "    return None"
        ),
        "Unclosed Resource (I/O)": (
            "def parse_system_logs(filename):\n"
            "    file_obj = open(filename, 'r')\n"
            "    entries = file_obj.readlines()\n"
            "    return [line.strip() for line in entries if 'ERROR' in line]"
        ),
        "Unbounded Recursion": (
            "def compute_fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)"
        )
    }

# 4. Main Clean Header
st.markdown('<div class="main-title">Autonomous Code Analysis Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Orchestrated with LangGraph | Model Context Protocol (MCP) | Gemini 3.1 Flash Lite'
    '</div>', 
    unsafe_allow_html=True
)

# 5. Multi-Tab Navigation (4 Dedicated Tabs)
tab_review, tab_tests, tab_journey, tab_capabilities = st.tabs([
    "Code Review & Diff View",
    "PyTest Suite Generator",
    "Engineering Architecture",
    "Capabilities & Guide"
])

# Helper function to extract Python blocks from markdown response
def extract_python_code(markdown_text: str) -> str:
    matches = re.findall(r"```python(.*?)```", markdown_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""

# ==============================================================================
# TAB 1: CODE REVIEW & DIFF VIEW
# ==============================================================================
with tab_review:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Source Code & Scope")
        
        default_objective = "Evaluate security vulnerabilities, PEP 8 compliance, performance, and architecture."
        user_task = st.text_input("Review Scope:", value=default_objective)
        
        initial_code = sample_codes.get(preset_choice, "") if preset_choice != "Custom Code" else ""
        
        user_code = st.text_area(
            "Paste Python Snippet:",
            value=initial_code,
            height=340,
            placeholder="def target_routine():\n    pass",
            key="review_code_input"
        )
        
        submit_button = st.button("Execute Code Review", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Senior Engineer Evaluation")
        
        if submit_button:
            if not user_code.strip():
                st.warning("Please provide a valid code snippet before initiating analysis.")
            else:
                with st.spinner("Executing AST inspection, OWASP security scanning, and architectural evaluation..."):
                    audit_directives = []
                    if enable_owasp:
                        audit_directives.append("- Conduct an OWASP Top 10 security audit (SQL injection, hardcoded secrets, input sanitization).")
                    if enable_pep8:
                        audit_directives.append("- Enforce strict PEP 8 naming conventions, explicit Python type hints (typing), and docstrings.")
                    
                    directives_text = "\n".join(audit_directives)

                    formatted_prompt = (
                        f"Review Objective: {user_task}\n\n"
                        f"Target Code:\n```python\n{user_code}\n```\n\n"
                        f"Special Audit Directives:\n{directives_text}\n\n"
                        f"Instructions:\n"
                        f"Provide a comprehensive, senior software engineer review in clean Markdown.\n"
                        f"Structure your response under these exact headings:\n"
                        f"### 1. Functional Correctness & Bug Hazards\n"
                        f"### 2. OWASP Security Vulnerability Audit\n"
                        f"### 3. PEP 8 & Code Style Compliance\n"
                        f"### 4. Performance & Resource Management\n"
                        f"### 5. Production-Ready Refactored Solution (Must wrap code inside standard ```python ``` block)\n"
                        f"### 6. Summary of Key Improvements"
                    )

                    event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(event_loop)
                    try:
                        evaluation_response = event_loop.run_until_complete(
                            run_agent_query(formatted_prompt, session_id=session_id)
                        )
                        st.session_state["last_review"] = evaluation_response
                        st.session_state["original_code"] = user_code
                        st.markdown(evaluation_response)
                    except Exception as error:
                        st.error(f"Execution Error: {str(error)}")
                    finally:
                        event_loop.close()
        elif "last_review" in st.session_state:
            st.markdown(st.session_state["last_review"])
        else:
            st.info("Input a code snippet on the left panel and click 'Execute Code Review' to inspect the report.")

    # Side-by-Side Diff View Section
    if "last_review" in st.session_state and "original_code" in st.session_state:
        st.markdown("---")
        st.subheader("Side-by-Side Code Comparison (Diff View)")
        
        extracted_refactored = extract_python_code(st.session_state["last_review"])
        
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.markdown("**Original Input Code**")
            st.code(st.session_state["original_code"], language="python")
            
        with diff_col2:
            st.markdown("**Production-Ready Refactored Code**")
            if extracted_refactored:
                st.code(extracted_refactored, language="python")
            else:
                st.info("Refactored code block could not be isolated for diff view.")

# ==============================================================================
# TAB 2: PYTEST SUITE GENERATOR
# ==============================================================================
with tab_tests:
    st.subheader("Automated Unit Test Suite Generator")
    st.write("Generates comprehensive, production-grade PyTest suites including happy paths, boundary cases, and exception handling.")
    st.markdown("---")
    
    test_col_in, test_col_out = st.columns([1, 1], gap="large")
    
    with test_col_in:
        st.markdown("**Target Function / Code for Testing**")
        
        # Preload with code from review tab if available
        default_test_target = st.session_state.get("original_code", "")
        test_target_code = st.text_area(
            "Paste Python Code to Test:",
            value=default_test_target,
            height=340,
            placeholder="def function_to_test():\n    pass",
            key="test_code_input"
        )
        
        test_framework = st.selectbox("Testing Framework:", ["PyTest with Mocks & Parametrize", "Standard unittest"])
        generate_test_btn = st.button("Generate Test Suite", type="primary", use_container_width=True)

    with test_col_out:
        st.markdown("**Generated Unit Test Suite**")
        
        if generate_test_btn:
            if not test_target_code.strip():
                st.warning("Please provide target code to generate tests for.")
            else:
                with st.spinner("Analyzing functions, inferring edge cases, and constructing PyTest suite..."):
                    test_prompt = (
                        f"Framework Requested: {test_framework}\n\n"
                        f"Target Code to Test:\n```python\n{test_target_code}\n```\n\n"
                        f"Instructions:\n"
                        f"Act as a Principal Software Engineer specialized in Quality Engineering.\n"
                        f"Generate an exhaustive, modular test suite covering:\n"
                        f"1. Standard Happy Path executions.\n"
                        f"2. Edge cases (empty structures, None values, boundaries).\n"
                        f"3. Error conditions and expected exception assertions (`pytest.raises`).\n"
                        f"4. Mocking external calls (e.g. database cursors, network I/O) using `unittest.mock`.\n"
                        f"Wrap the full runnable test code inside a single ```python ``` block with clear explanatory comments."
                    )
                    
                    event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(event_loop)
                    try:
                        test_response = event_loop.run_until_complete(
                            run_agent_query(test_prompt, session_id=f"{session_id}_tests")
                        )
                        st.markdown(test_response)
                    except Exception as error:
                        st.error(f"Execution Error: {str(error)}")
                    finally:
                        event_loop.close()
        else:
            st.info("Input a Python function or snippet on the left and click 'Generate Test Suite' to produce automated tests.")

# ==============================================================================
# TAB 3: ARCHITECTURE & ROADMAP
# ==============================================================================
with tab_journey:
    st.subheader("System Architecture & Development Roadmap")
    st.write("A breakdown of the technical phases implemented to construct this autonomous code intelligence system.")
    st.markdown("---")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 01</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Lexical BM25 Search Engine</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Constructed an AST-based tokenizer to index classes, functions, and docstrings '
            'from the <code>requests</code> repository using Okapi BM25 for deterministic keyword matching.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_p2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 02</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Semantic Dense Embeddings</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Generated 384-dimensional vector embeddings via <code>all-MiniLM-L6-v2</code>. '
            'Enables semantic code discovery when queries differ from function or variable names.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col_p3, col_p4 = st.columns(2)
    with col_p3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 03</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Hybrid Fusion & Reranking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Combined lexical and vector outputs using Reciprocal Rank Fusion (RRF). '
            'Applied <code>cross-encoder/ms-marco-MiniLM-L-6-v2</code> to eliminate retrieval noise.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_p4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 04</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">RAG Benchmarking (DeepEval)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Implemented LLM-as-a-Judge test suites measuring Faithfulness, Answer Relevancy, '
            'and Contextual Precision against ground-truth codebase questions.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col_p5, col_p6 = st.columns(2)
    with col_p5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 05</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Model Context Protocol (MCP)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Wrapped the hybrid search pipeline into an MCP stdio server. '
            'Exposed the local codebase search tool over standardized JSON-RPC protocols.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_p6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<span class="phase-badge">PHASE 06</span>', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">State Graph & Docker Container</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-desc">'
            'Built cyclic agent state machines with async SQLite checkpoints (<code>agent_memory.db</code>). '
            'Containerized the pipeline with model pre-caching and volume mounts.'
            '</div>', 
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: CAPABILITIES & VALUE
# ==============================================================================
with tab_capabilities:
    st.subheader("Functional Capabilities & Use Cases")
    st.write("Technical audit capabilities provided by the agent across the development lifecycle.")
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### OWASP Security Audit")
        st.markdown(
            "- **Injection Defense:** Pinpoints SQL, command, and LDAP injection risks caused by dynamic string formatting.\n"
            "- **Credential Leakage:** Detects hardcoded tokens, passwords, and sensitive API keys in the source tree.\n"
            "- **Data Sanitization:** Recommends parameterized queries and input validation boundaries."
        )

    with c2:
        st.markdown("#### PEP 8 & Clean Architecture")
        st.markdown(
            "- **Type Completeness:** Enforces explicit type annotations (`typing.List`, `Dict`, `Optional`).\n"
            "- **Code Style Standards:** Flags non-standard naming conventions and formatting violations.\n"
            "- **Defensive Programming:** Recommends context managers and explicit error hierarchies."
        )

    with c3:
        st.markdown("#### Automated Testing & Verification")
        st.markdown(
            "- **PyTest Generation:** Generates test suites with assertions and parameterized inputs.\n"
            "- **Mocking:** Sets up mock fixtures for network calls, file handles, and database connections.\n"
            "- **Regression Defense:** Identifies edge cases before code merges to production."
        )