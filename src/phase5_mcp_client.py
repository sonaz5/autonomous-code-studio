"""
Phase 5 -- MCP Client Implementation (Model Context Protocol 2.x)

Goal: Connect an autonomous LangGraph agent to the Phase 5 MCP Server over stdio.
The client dynamically discovers the 'search_codebase' tool, manages persistent async SQLite
state via AsyncSqliteSaver, and streams structured senior-level code reviews.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Execution paths for subprocess invocation
PYTHON_EXE = sys.executable
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "phase5_mcpserver.py")


class AgentState(TypedDict):
    """The graph state schema holding message history."""
    messages: Annotated[list[BaseMessage], add_messages]


def extract_clean_text(content: object) -> str:
    """Extracts raw text from message content whether it is a string or list of dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                extracted.append(block["text"])
            elif isinstance(block, str):
                extracted.append(block)
        return "".join(extracted)
    return str(content)


async def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        return

    if not os.path.exists(SERVER_SCRIPT):
        print(f"Error: Target MCP Server script not found at {SERVER_SCRIPT}", file=sys.stderr)
        return

    print("=" * 60)
    print("Phase 5: LangGraph Senior Engineer Agent (MCP Client)")
    print("=" * 60)
    print("Starting and connecting to MCP Server over stdio...")

    server_params = StdioServerParameters(
        command=PYTHON_EXE,
        args=[SERVER_SCRIPT],
        env=os.environ.copy()
    )

    # Establish connection with the MCP server subprocess
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Dynamic tool discovery
            tools_response = await session.list_tools()
            available_tools = tools_response.tools
            print(f"Discovered {len(available_tools)} MCP tool(s):")
            for t in available_tools:
                print(f" - {t.name}: {t.description.strip()}")

            # Wrap MCP call as LangChain Tool
            @tool
            async def search_codebase(query: str) -> str:
                """Searches the target repository for functions, classes, and security logic."""
                print(f"\n[MCP Stdio Invoke] Executing 'search_codebase' with query: '{query}'...")
                response = await session.call_tool("search_codebase", arguments={"query": query})
                
                text_chunks = [
                    item.text for item in response.content if getattr(item, "type", None) == "text"
                ]
                return "\n".join(text_chunks) if text_chunks else "No relevant code found."

            bound_tools = [search_codebase]

            # LLM Configuration (using flash-lite to prevent 429 quota exhaustion)
            llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
            llm_with_tools = llm.bind_tools(bound_tools)

            def chatbot_node(state: AgentState):
                system_instruction = SystemMessage(
                    content=(
                        "You are a strict Senior Software Engineer. You have access to a codebase search tool via MCP. "
                        "Always search relevant files before answering or performing code reviews. "
                        "Structure your answers with Complexity, Security, and Maintainability sections."
                    )
                )
                payload = [system_instruction] + state["messages"]
                return {"messages": [llm_with_tools.invoke(payload)]}

            # Graph Assembly
            workflow = StateGraph(AgentState)
            workflow.add_node("chatbot", chatbot_node)
            workflow.add_node("tools", ToolNode(tools=bound_tools))
            workflow.add_conditional_edges("chatbot", tools_condition)
            workflow.add_edge("tools", "chatbot")
            workflow.add_edge(START, "chatbot")

            # Persistent Async SQLite memory setup
            db_path = "agent_memory.db"
            async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
                app = workflow.compile(checkpointer=checkpointer)
                session_config = {"configurable": {"thread_id": "phase5_master_session"}}

                print("\nAgent initialized and ready! Enter your query (type 'exit' to quit).\n")

                while True:
                    user_query = input(">>> ").strip()
                    if user_query.lower() in ["exit", "quit"]:
                        print("Terminating Phase 5 Client session. Goodbye!")
                        break

                    if not user_query:
                        continue

                    print("\nProcessing request via MCP Pipeline...")
                    try:
                        async for event in app.astream(
                            {"messages": [HumanMessage(content=user_query)]},
                            session_config,
                            stream_mode="updates"
                        ):
                            if "chatbot" in event:
                                message = event["chatbot"]["messages"][-1]
                                if not getattr(message, "tool_calls", None) and message.content:
                                    clean_output = extract_clean_text(message.content)
                                    print(f"\n--- Senior Engineer Review ---\n{clean_output}\n" + "-" * 30)

                    except Exception as err:
                        err_text = str(err)
                        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                            print("\n[Quota Error] API quota reached. Please wait ~60s before retrying.")
                        else:
                            print(f"\n[Runtime Error] {err_text}")


if __name__ == "__main__":
    asyncio.run(main())