"""One Deep Agent implemented with LangGraph.

This module builds an iterative, tool-using agent that can:
1. Reason about a task.
2. Call tools.
3. Reflect on intermediate outputs.
4. Stop when an answer is ready or a depth limit is reached.

Usage:
    export OPENAI_API_KEY=...  # required
    python one_deep_agent.py "Research the impact of transformers in NLP"
"""

from __future__ import annotations

import argparse
import json
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely.

    Example inputs: "(12 + 7) * 3", "4**3", "100 / 8"
    """
    allowed = set("0123456789+-*/(). %")
    if not set(expression).issubset(allowed):
        return "Error: expression contains unsupported characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


@tool
def json_pretty(data: str) -> str:
    """Pretty-print JSON input as indented text."""
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2, sort_keys=True)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"


TOOLS = [calculator, json_pretty]


class AgentState(TypedDict):
    """State flowing through the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    max_iterations: int


def build_agent_graph(model_name: str = "gpt-4o-mini"):
    """Create a One-Deep-style reflective agent with bounded depth."""
    llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    system_prompt = (
        "You are One Deep Agent: a focused reasoning assistant. "
        "Think step-by-step, use tools when useful, and keep answers concise. "
        "After each tool result, reflect and decide whether another tool call is necessary. "
        "If enough information is present, give a final answer."
    )

    def think(state: AgentState) -> dict[str, Any]:
        msgs = [HumanMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(msgs)
        return {"messages": [response], "iterations": state["iterations"] + 1}

    def route(state: AgentState) -> Literal["tools", "final", "maxed"]:
        last = state["messages"][-1]
        if state["iterations"] >= state["max_iterations"]:
            return "maxed"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "final"

    def maxed_out_message(state: AgentState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I reached the maximum reasoning depth before finalizing. "
                        "Please increase max_iterations or simplify the task."
                    )
                )
            ]
        }

    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("tools", tool_node)
    graph.add_node("maxed", maxed_out_message)

    graph.set_entry_point("think")
    graph.add_conditional_edges(
        "think",
        route,
        {
            "tools": "tools",
            "final": END,
            "maxed": "maxed",
        },
    )
    graph.add_edge("tools", "think")
    graph.add_edge("maxed", END)

    return graph.compile()


def run(query: str, max_iterations: int = 6, model_name: str = "gpt-4o-mini") -> str:
    """Run the One Deep Agent with a user query."""
    app = build_agent_graph(model_name=model_name)
    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "iterations": 0,
            "max_iterations": max_iterations,
        }
    )

    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
        if isinstance(msg, ToolMessage):
            continue
    return "No response produced."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run One Deep Agent (LangGraph)")
    parser.add_argument("query", help="User question/task for the agent")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum think/tool loops before forced stop",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Chat model name supported by langchain_openai",
    )

    args = parser.parse_args()
    answer = run(args.query, max_iterations=args.max_iterations, model_name=args.model)
    print(answer)


if __name__ == "__main__":
    main()
