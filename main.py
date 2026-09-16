"""
Main Orchestration Pipeline for Autonomous Code Analysis Agent.
Integrates LangGraph, Gemini 3.1 Flash Lite, and Async SQLite Checkpointing.
Supports both programmatic execution (Streamlit UI) and interactive CLI.
"""

import os
import asyncio
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Path to the persistent SQLite database
DB_PATH = "agent_memory.db"


async def run_agent_query(user_query: str, session_id: str = "default_session") -> str:
    """
    Executes a single evaluation query within an isolated LangGraph workflow.
    Safely extracts plain text from either string or block-list responses.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please define it in your PowerShell session.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )

    workflow = StateGraph(MessagesState)

    async def call_model(state: MessagesState):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}

        response = await app.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config=config
        )
        
        raw_content = response["messages"][-1].content

    
        if isinstance(raw_content, list):
            extracted_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
            return extracted_text
        elif isinstance(raw_content, dict):
            return raw_content.get("text", str(raw_content))
        
        return str(raw_content)

async def run_cli():
    """Fallback interactive CLI loop for terminal testing."""
    print("Master Project Assembly: Autonomous Code Analysis Agent")
    print("=" * 65)
    print("All systems operational. Type your query or 'exit' to quit.\n")

    session_id = "cli_session"

    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip().lower() == "exit":
                print("Terminating agent pipeline. State preserved in SQLite.")
                break
            if not user_input.strip():
                continue

            output = await run_agent_query(user_input, session_id=session_id)
            print(f"\n--- Senior Engineer Review ---\n{output}\n------------------------------\n")
        except Exception as error:
            print(f"Execution Error: {error}")


if __name__ == "__main__":
    asyncio.run(run_cli())