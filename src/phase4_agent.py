"""
Phase 4 -- Agentic Tool Calling with Persistent SQLite Memory

Goal: Create an autonomous LangGraph agent that uses the Phase 3 Hybrid RAG 
engine as a tool to search a local codebase. The agent's memory is saved 
persistently to an SQLite database, ensuring conversations survive restarts.
"""

import os
import glob
import sys
import sqlite3
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# Import the RAG engine we built in Phase 3
# Ensure phase3_hybrid_rag.py is in the same directory
from phase3_hybrid_rag import HybridIndex, Reranker, chunk_python_file, hybrid_search

# --- 1. Global State for RAG Initialization ---
GLOBAL_INDEX = None
GLOBAL_RERANKER = None

# --- 2. Tool Definition ---
@tool
def search_codebase(query: str) -> str:
    """
    Searches the local codebase for functions and classes relevant to the query.
    Always use this tool to find context, definitions, or security practices in the codebase 
    before answering questions about the project or writing a code review.
    """
    print(f"\n[Agent Tool Execution] Searching codebase for: '{query}'...")
    if not GLOBAL_INDEX or not GLOBAL_RERANKER:
        return "Error: Search index is not initialized."
        
    results = hybrid_search(GLOBAL_INDEX, GLOBAL_RERANKER, query, retrieve_k=10, final_k=3)
    
    if not results:
        return "No relevant code found in the codebase."
        
    formatted_results = []
    for chunk, score in results:
        formatted_results.append(
            f"--- File: {chunk['file']} (Line {chunk['start_line']}) ---\n{chunk['text']}\n"
        )
        
    return "\n".join(formatted_results)

# --- 3. LangGraph Setup ---
class AgentState(TypedDict):
    """The state of our agent, simply a list of messages."""
    messages: Annotated[list[BaseMessage], add_messages]

def build_graph(memory: SqliteSaver):
    """Builds and compiles the LangGraph architecture using persistent memory."""
    
    # CHANGED: Switched to gemini-3.1-flash-lite to bypass the 429 quota block
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
    
    tools = [search_codebase]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot_node(state: AgentState):
        """The main decision-making node for the agent."""
        system_prompt = SystemMessage(
            content="You are a strict, Senior Software Engineer. You have a tool to search the codebase. "
                    "Always use it to look up relevant files before providing a code review or answering questions. "
                    "Structure your reviews professionally."
        )
        
        messages_to_send = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(messages_to_send)
        return {"messages": [response]}

    # Initialize Graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot_node)
    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Routing logic
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    return graph_builder.compile(checkpointer=memory)

# --- 4. System Initialization & CLI ---
def initialize_rag_system(folder_path: str):
    """Initializes the Hybrid RAG index for the given folder."""
    global GLOBAL_INDEX, GLOBAL_RERANKER
    
    py_files = glob.glob(f"{folder_path}/**/*.py", recursive=True)
    if not py_files:
        print(f"Error: No .py files found under {folder_path}")
        sys.exit(1)

    print(f"Building local search index from {len(py_files)} files... Please wait.")
    all_chunks = []
    for path in py_files:
        all_chunks.extend(chunk_python_file(path))

    if not all_chunks:
        print("Error: No chunks extracted.")
        sys.exit(1)

    GLOBAL_INDEX = HybridIndex()
    GLOBAL_INDEX.build(all_chunks)
    GLOBAL_RERANKER = Reranker()
    print("Search index and reranker successfully initialized!")

def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        return

    target_folder = "../requests-main/src/requests"
    if not os.path.exists(target_folder):
        target_folder = "requests-main/src/requests"
        if not os.path.exists(target_folder):
            target_folder = "."

    print("=" * 60)
    print("Phase 4: Agentic Code Reviewer with SQLite Memory")
    print("=" * 60)
    
    initialize_rag_system(target_folder)
    
    db_path = "agent_memory.db"
    print(f"Connecting to persistent memory database at: {db_path}")
    
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        app = build_graph(memory)
        
        config = {"configurable": {"thread_id": "senior_dev_session_1"}}
        
        print("\nAgent is ready! Ask a codebase question or request a code review.")
        print("Type 'exit' to quit.\n")
        
        # Chat Loop with Graceful Error Handling
        while True:
            user_input = input(">>> ")
            if user_input.lower() in ["exit", "quit"]:
                print("Conversation saved. Goodbye!")
                break
                
            print("\nAgent is thinking (and searching if needed)...")
            
            try:
                events = app.stream(
                    {"messages": [HumanMessage(content=user_input)]}, 
                    config, 
                    stream_mode="updates" 
                )
                
                for event in events:
                    if "chatbot" in event:
                        last_message = event["chatbot"]["messages"][-1]
                        
                        # Check if it is not a tool call and has content
                        if not getattr(last_message, 'tool_calls', None) and last_message.content:
                            
                            # CLEAN OUTPUT PARSER: Extract raw text if LangChain returns a list of dicts
                            raw_content = last_message.content
                            if isinstance(raw_content, list):
                                clean_text = "".join(
                                    [item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item]
                                )
                            else:
                                clean_text = str(raw_content)
                                
                            print(f"\n--- Agent Response ---\n{clean_text}\n----------------------")
                            
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print("\n[Quota Error] You hit the Gemini Free Tier limit.")
                    print("Please wait about 60 seconds, then try asking your question again.")
                else:
                    print(f"\n[Unexpected Error] {error_msg}")

if __name__ == "__main__":
    main()