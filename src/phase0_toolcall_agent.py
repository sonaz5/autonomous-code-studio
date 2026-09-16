"""
Phase 0 -- Manual Tool-Calling Loop (Google Gemini free tier)

Goal: understand what LangGraph does under the hood BEFORE using it.
This script implements the raw function-call loop directly with the
official google-genai SDK. No LangChain, no LangGraph -- just the
mechanics.

Get a free API key at:
    https://aistudio.google.com/apikey

Usage:
    export GEMINI_API_KEY="your_key_here"  # PowerShell: $env:GEMINI_API_KEY = "your_key_here"
    python src/phase0_toolcall_agent.py "What is 23 * 47, and what's the weather in Istanbul?"
"""

import os
import sys

from google import genai
from google.genai import types

# Gemini's free tier is Flash-only. Check https://aistudio.google.com for the
# current model list if this identifier ever stops working.
MODEL = "gemini-3.6-flash"

# ---- Tool implementations --------------------------------------------------


def get_weather(city: str) -> str:
    """Mock weather lookup. Swap this for a real API call later."""
    fake_data = {
        "istanbul": "18C, partly cloudy",
        "san francisco": "15C, foggy",
        "berlin": "12C, rain",
    }
    return fake_data.get(city.lower(), f"No data for '{city}'")


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely (no builtins, whitelisted chars)."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


TOOL_FUNCTIONS = {"get_weather": get_weather, "calculator": calculator}

# Gemini's native tool schema: a Tool wrapping one or more function declarations.
TOOLS = [
    types.Tool(
        function_declarations=[
            {
                "name": "get_weather",
                "description": "Get current weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression, e.g. '23*47'.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        ]
    )
]


def run_tool(name: str, tool_input: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    return fn(**tool_input)


# ---- The agent loop itself --------------------------------------------------


def run_agent(user_message: str, max_turns: int = 5) -> str:
    """
    This is the loop every agent framework (LangGraph included) wraps in
    higher-level abstractions. Seeing it raw once makes the framework's
    state graph make a lot more sense later.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(tools=TOOLS)

    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    for turn in range(max_turns):
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)  # model's turn goes back into history

        function_calls = [
            part.function_call for part in candidate.content.parts if part.function_call
        ]

        if not function_calls:
            # Model gave a final text answer -- we're done.
            return response.text

        # Model wants to call one or more tools. Execute each, then report
        # the results back in a single "user" turn before continuing.
        function_response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            print(f"[tool call] {fc.name}({args})", file=sys.stderr)
            result = run_tool(fc.name, args)
            function_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

    return "Max turns reached without a final answer."


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is 23 * 47?"
    print(run_agent(query))
