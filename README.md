# One Deep Agent (LangGraph)

This repository now includes a "One Deep Agent" implementation using **LangGraph**.

## Files

- `one_deep_agent.py`: iterative deep-reasoning agent graph with tool use and bounded depth.
- `requirements.txt`: Python dependencies.

## What this agent does

The agent loops through these steps:

1. **Think**: model reasons on the current conversation state.
2. **Act (optional)**: if tool calls are emitted, execute tools.
3. **Reflect**: re-enter think step with tool outputs.
4. **Stop**: return final answer, or halt when max depth is reached.

Included tools:

- `calculator(expression)`: evaluates basic arithmetic.
- `json_pretty(data)`: pretty-prints JSON.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
```

## Run

```bash
python one_deep_agent.py "What is (24+18)*3?"
python one_deep_agent.py "Format this JSON: {\"z\":1,\"a\":2}" --max-iterations 8
```

## Notes

- Default model: `gpt-4o-mini` (change with `--model`).
- Depth limit defaults to 6 loops (`--max-iterations`).
