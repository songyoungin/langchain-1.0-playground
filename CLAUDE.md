# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain 1.0 playground - production-ready examples for testing and learning LangChain 1.0 API patterns.

## Environment Setup

- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: uv (uses `uv.lock` for dependency locking)
- **Virtual Environment**: `.venv` in project root

## Essential Commands

### Package Management
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>
```

### Running Examples
```bash
# Activate virtual environment and run an example
source .venv/bin/activate
python examples/create_agent.py
```

## Code Architecture

### LangChain 1.0 API Patterns

This project uses LangChain 1.0+ APIs. Key patterns:

**Agent Creation:**
```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[tool_function],
    system_prompt="System instructions..."
)
```

**Tool Definition:**
- Tools are plain Python functions with type hints and docstrings
- Docstring format is critical: must include Args and Returns sections
- Type hints are required for tool parameters

**Agent Invocation:**
```python
inputs = {"messages": [{"role": "user", "content": "query"}]}
result = agent.invoke(inputs)
```

### Project Structure

- `main.py`: Entry point
- `examples/`: Contains working examples demonstrating LangChain 1.0 patterns
  - `create_agent.py`: Basic agent creation with custom tools

## Key Notes

- When creating new examples, follow the patterns in `examples/create_agent.py`
- Tool functions must have complete docstrings with Args and Returns sections for LangChain to properly parse them
- Use the `invoke()` method for agent calls, passing messages in the format `{"messages": [...]}`
