import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver 

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely. Example: '23 * 47' or '23 x 47'."""
    

    expression = expression.lower().replace("x", "*")
    
    allowed_chars = set("0123456789+-*/(). ")
    
    if not set(expression) <= allowed_chars:
        return "Error: Expression contains disallowed characters. Please use only numbers and basic operators (+, -, *, /, x)."
    
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error occurred during calculation: {exc}. Please correct the expression and try again."

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    fake_data = {
        "istanbul": "18C, partly cloudy",
        "san francisco": "15C, foggy",
        "berlin": "12C, rain",
    }
    
    result = fake_data.get(city.lower())
    if result:
        return result
    else:
        return f"Error: No weather data found for '{city}'. Please check the city name and try again."

tools = [calculator, get_weather]
tool_node = ToolNode(tools)


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash" 
).bind_tools(tools)

def call_model(state: State):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set the GEMINI_API_KEY environment variable.")
        exit(1)

   
    user_prompt = "What is 23 x 47, and what's the weather in Istanbul?"
    print(f"User: {user_prompt}\n")
    print("-" * 40)
    
    inputs = {"messages": [("user", user_prompt)]}
    

    config = {"configurable": {"thread_id": "ornek_sohbet_1"}, "recursion_limit": 50}
    
    for event in app.stream(inputs, config=config):
        for node_name, node_state in event.items():
            print(f"--- Node [{node_name}] Executed ---")
            
            last_message = node_state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"Action: Calling {tool_call['name']} with args {tool_call['args']}")
            
            elif node_name == "tools":
                 print(f"Tool Result: {last_message.content}")
                 
    print("-" * 40)
    
    # Retrieve the final state using our memory checkpointer
    final_state = app.get_state(config)
    final_message = final_state.values["messages"][-1]
    
    # Safely extract the text from the content
    content = final_message.content
    if isinstance(content, list):
        # If the content is a list of dictionaries, extract the 'text' from each dictionary
        clean_text = "".join([part.get("text", "") for part in content if isinstance(part, dict) and "text" in part])
    else:
        # If it is already a simple string, just use it directly
        clean_text = content

    print(f"\nFinal Agent Response:\n{clean_text}")