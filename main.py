"""
Main Orchestration Pipeline for Autonomous Code Analysis Agent.

Integrates LangGraph, Gemini, a Hybrid-RAG `search_codebase` tool (see
phase3_hybrid_rag.py), and Async SQLite checkpointing. Supports both
programmatic execution (Streamlit UI) and interactive CLI.

Architecture note -- why this calls phase3 directly instead of going
through the phase5 MCP server/client:
Spawning an MCP stdio subprocess (phase5_mcpserver.py) from inside a
Streamlit app is fragile in a hosted environment (process lifecycle,
stdio buffering, no shell access on some deployment targets). Phase 4
already showed the simpler, equally valid alternative: call the same
HybridIndex/Reranker classes in-process, wrapped as a LangChain tool.
That's what happens here. phase5's MCP client/server remain a useful,
working demonstration of the protocol on their own -- this file just
doesn't route production traffic through them.
"""

import ast
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.phase3_hybrid_rag import EMBEDDING_MODEL, HybridIndex, Reranker, hybrid_search

# Path to the persistent SQLite database
DB_PATH = "agent_memory.db"

# Single source of truth for the model name used by the live app (the
# other phase*.py scripts each hardcode their own -- see the "kod tekrarı"
# cleanup item for consolidating those too).
MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = (
    "You are a strict Senior Software Engineer. You have a `search_codebase` "
    "tool that searches the code provided for this task (the snippet under "
    "review, plus any attached context files) for related functions or "
    "classes. Use it whenever the code calls a helper you haven't seen yet, "
    "or when you want to check whether a pattern (e.g. a security issue) "
    "repeats elsewhere in the provided code. Structure your response "
    "exactly as instructed in the user's message."
)


class AgentState(TypedDict):
    """The graph state schema holding message history."""
    messages: Annotated[list[BaseMessage], add_messages]


# --- Model caching -----------------------------------------------------------
# SentenceTransformer / CrossEncoder weights are only loaded once per process
# and reused across requests. Without this, build_codebase_tool() below would
# reload ~200MB of embedding + reranker weights from disk on every single
# button click in the Streamlit app, which is slow enough to make the tool
# feel broken even when it's working correctly.
_EMBEDDER = None
_RERANKER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER


def _get_reranker() -> Reranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER


# --- Chunking an in-memory snippet -------------------------------------------


def _chunk_source(label: str, source: str) -> list[dict]:
    """Same idea as phase3_hybrid_rag.chunk_python_file, but works on a
    string already in memory (what the user pasted into the UI) instead of
    reading a file from disk."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not fully valid Python (e.g. a deliberately partial snippet) --
        # fall back to one chunk so search still has something to work with.
        return [{"id": f"{label}:0", "file": label, "name": label,
                  "type": "Module", "start_line": 1, "text": source}]

    lines = source.splitlines()
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            chunks.append({
                "id": f"{label}:{node.name}:{node.lineno}",
                "file": label,
                "name": node.name,
                "type": type(node).__name__,
                "start_line": node.lineno,
                "text": "\n".join(lines[start:end]),
            })
    return chunks or [{"id": f"{label}:0", "file": label, "name": label,
                         "type": "Module", "start_line": 1, "text": source}]


def build_codebase_tool(context_files: dict[str, str]):
    """Builds a `search_codebase` tool scoped to whatever code was provided
    for THIS request -- the snippet under review plus any optional context
    files -- instead of a hardcoded external repo (what phase4/phase5 did).

    If nothing was provided, returns a tool that says so rather than
    silently indexing nothing or raising, so the agent gets a clear signal
    instead of a confusing empty result.
    """
    all_chunks: list[dict] = []
    for filename, source in (context_files or {}).items():
        if source and source.strip():
            all_chunks.extend(_chunk_source(filename, source))

    if not all_chunks:
        @tool
        async def search_codebase(query: str) -> str:
            """Search the code provided for this task for related functions or classes."""
            return "No code has been provided to search yet."
        return search_codebase

    index = HybridIndex(embedder=_get_embedder())
    index.build(all_chunks)
    reranker = _get_reranker()

    @tool
    async def search_codebase(query: str) -> str:
        """Search the code provided for this task (the snippet under review
        and any attached context files) for functions or classes relevant
        to the query. Use this to find how a function is used elsewhere, or
        to check a helper before judging it."""
        results = hybrid_search(index, reranker, query, retrieve_k=8, final_k=3)
        if not results:
            return "No relevant code found for that query."
        return "\n".join(
            f"--- {c['type']} {c['name']} ({c['file']}:{c['start_line']}) ---\n{c['text']}\n"
            for c, _ in results
        )

    return search_codebase


def _extract_text(raw_content) -> str:
    """Safely extracts plain text from either a string or a block-list response."""
    if isinstance(raw_content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )
    if isinstance(raw_content, dict):
        return raw_content.get("text", str(raw_content))
    return str(raw_content)


async def run_agent_query(
    user_query: str,
    session_id: str = "default_session",
    context_files: Optional[dict[str, str]] = None,
) -> tuple[str, list[dict]]:
    """
    Executes a single query within an isolated, tool-calling LangGraph
    workflow. The agent can call `search_codebase` (backed by the Phase 3
    Hybrid RAG engine, scoped to `context_files`) before answering.

    Returns (answer_text, tool_calls_made) so callers -- notably the
    Streamlit UI -- can show the user what the agent actually searched for,
    instead of the tool use happening invisibly.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please define it in your PowerShell session.")

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2, google_api_key=api_key)

    search_codebase = build_codebase_tool(context_files or {})
    tools = [search_codebase]
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState):
        messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages_to_send)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools=tools))
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.add_edge(START, "agent")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}

        response = await app.ainvoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
        )

        messages = response["messages"]

        # Walk the full message history for this turn to log every tool
        # call the agent made, so the UI can show it -- not just the final
        # answer.
        tool_calls_made: list[dict] = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                tool_calls_made.append({"name": call["name"], "args": call.get("args", {})})

        answer_text = _extract_text(messages[-1].content)
        return answer_text, tool_calls_made


async def run_cli():
    """Fallback interactive CLI loop for terminal testing."""
    print("Master Project Assembly: Autonomous Code Analysis Agent")
    print("=" * 65)
    print("All systems operational. Type your query or 'exit' to quit.\n")
    print("Tip: paste code directly in your message -- the agent can search")
    print("it with the search_codebase tool while forming its answer.\n")

    session_id = "cli_session"

    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip().lower() == "exit":
                print("Terminating agent pipeline. State preserved in SQLite.")
                break
            if not user_input.strip():
                continue

            output, tool_calls = await run_agent_query(user_input, session_id=session_id)
            if tool_calls:
                print("\n[Tool calls made]")
                for call in tool_calls:
                    print(f"  - {call['name']}({call['args']})")
            print(f"\n--- Senior Engineer Review ---\n{output}\n------------------------------\n")
        except Exception as error:
            print(f"Execution Error: {error}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_cli())