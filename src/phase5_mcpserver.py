"""
Phase 5 -- MCP Server Implementation (Model Context Protocol)

Goal: Expose the Phase 4 Hybrid RAG engine as a standardized MCP Tool over stdio.
Any MCP-compliant client (LangGraph, Claude Desktop, Cursor) can connect to this
server and search the target codebase.
"""

import os
import glob
import sys
from mcp.server.fastmcp import FastMCP

# Import the existing Hybrid RAG engine from Phase 4
from phase3_hybrid_rag import HybridIndex, Reranker, chunk_python_file, hybrid_search

# --- 1. Initialize FastMCP Server ---
mcp = FastMCP("Codebase-Search-Server")

# Global variables to hold the RAG engine in memory
GLOBAL_INDEX = None
GLOBAL_RERANKER = None


def setup_rag_engine(folder_path: str):
    """Indexes the target folder using the Phase 4 Hybrid RAG architecture."""
    global GLOBAL_INDEX, GLOBAL_RERANKER

    py_files = glob.glob(f"{folder_path}/**/*.py", recursive=True)
    if not py_files:
        print(f"Error: No .py files found under {folder_path}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    for path in py_files:
        all_chunks.extend(chunk_python_file(path))

    if not all_chunks:
        print("Error: No chunks extracted.", file=sys.stderr)
        sys.exit(1)

    GLOBAL_INDEX = HybridIndex()
    GLOBAL_INDEX.build(all_chunks)
    GLOBAL_RERANKER = Reranker()


# --- 2. Define MCP Tool ---
@mcp.tool()
def search_codebase(query: str) -> str:
    """Searches the local codebase for functions and classes relevant to the query.

    Use this tool to find context, definitions, and security logic before answering questions.
    """
    if not GLOBAL_INDEX or not GLOBAL_RERANKER:
        return "Error: Search index is not initialized."

    results = hybrid_search(
        GLOBAL_INDEX, GLOBAL_RERANKER, query, retrieve_k=10, final_k=3
    )
    if not results:
        return "No relevant code found."

    formatted_results = []
    for chunk, score in results:
        formatted_results.append(
            f"--- File: {chunk['file']} (Line {chunk['start_line']}) ---\n{chunk['text']}\n"
        )

    return "\n".join(formatted_results)


# --- 3. Server Startup ---
if __name__ == "__main__":
    # Target folder: requests library in the project root
    target_folder = "requests-main/src/requests"
    if not os.path.exists(target_folder):
        target_folder = "../requests-main/src/requests"
        if not os.path.exists(target_folder):
            target_folder = "."

    # Build the search index before starting the stdio loop
    setup_rag_engine(target_folder)

    # Run the server over stdio
    mcp.run(transport="stdio")