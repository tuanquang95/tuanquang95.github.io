---
title: 'Multi agent with LangChain'
date: 2026-01-01
permalink: /posts/2026/01/multi-agent-with-langchain/
tags:
  - multi-agent
  - langchain
  - LLMOPS
  - AI Engineering
---

# Building Multi-Agent Systems with LangChain: A Production-Ready Guide

---

## Introduction

As LLM applications grow in complexity, single-agent architectures start to hit their limits. Tasks that require planning, tool use, memory, and collaboration across different domains demand something more powerful — **multi-agent systems**.

In this post, we'll walk through how to build a robust multi-agent pipeline using **LangChain** and **LangGraph**, covering everything from the core concepts to production patterns you can deploy today.

---

## What Is a Multi-Agent System?

A **multi-agent system (MAS)** is an architecture where multiple autonomous LLM-powered agents collaborate to solve complex tasks. Each agent has:

- A **role** (e.g., Researcher, Writer, Critic)
- Access to specific **tools** (web search, code execution, databases)
- Its own **memory** and **context**
- A defined **communication protocol** with other agents

Think of it like a team of specialists — instead of one generalist trying to do everything, you delegate work to the right expert.

```
User Request
     │
     ▼
┌─────────────┐
│  Supervisor │  ← Orchestrates and delegates
│    Agent    │
└──────┬──────┘
       │
  ┌────┴─────┬──────────┐
  ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐
│ RAG  │  │Code  │  │Write │
│Agent │  │Agent │  │Agent │
└──────┘  └──────┘  └──────┘
```

---

## Why LangChain + LangGraph?

**LangChain** provides the building blocks:
- Standardized LLM interfaces
- Tool and retriever abstractions
- Memory and prompt management

**LangGraph** extends LangChain with:
- Stateful, graph-based agent orchestration
- Conditional routing between nodes
- Built-in support for cycles, checkpoints, and human-in-the-loop

Together, they give you everything needed to build production-grade multi-agent pipelines.

---

## Core Concepts Before We Build

### 1. Agent Node
Each agent in LangGraph is a **node** in a directed graph. A node receives state, processes it (via LLM + tools), and returns updated state.

### 2. Edges & Routing
**Edges** connect nodes. You can define:
- **Static edges** — always go from A → B
- **Conditional edges** — route dynamically based on the agent's output

### 3. Shared State
All agents share a **state object** — a typed dictionary passed through the graph. This is how agents communicate with each other.

### 4. Supervisor Pattern
A **Supervisor Agent** is responsible for:
- Receiving the user's task
- Deciding which agent to call next
- Aggregating final results

---

## Project Setup

```bash
pip install langchain langgraph langchain-openai langchain-community
```

```python
# .env
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here  # For web search tool
```

---

## Step 1: Define the Shared State

```python
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    task: str
    research_output: str
    final_output: str
```

The `add_messages` annotation ensures messages are **appended**, not overwritten, as they flow through the graph.

---

## Step 2: Create the Tools

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Web search tool
search_tool = TavilySearchResults(max_results=5)

# Custom code execution tool
@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code)
    return output.getvalue()
```

---

## Step 3: Build the Individual Agents

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Research Agent ---
research_agent = create_react_agent(
    llm,
    tools=[search_tool],
    state_modifier=(
        "You are a Research Agent. Your job is to gather accurate, "
        "up-to-date information on the given topic using web search. "
        "Be thorough and cite your sources."
    )
)

# --- Code Agent ---
code_agent = create_react_agent(
    llm,
    tools=[run_python],
    state_modifier=(
        "You are a Code Agent. You write clean, efficient Python code "
        "to solve analytical or data processing tasks. Always test your code."
    )
)

# --- Writer Agent ---
writer_agent = create_react_agent(
    llm,
    tools=[],
    state_modifier=(
        "You are a Writer Agent. Given research and data, you produce "
        "clear, concise, and well-structured written content for technical audiences."
    )
)
```

---

## Step 4: Build the Supervisor

The Supervisor is the brain of the system — it reads the current state and decides which agent to invoke next.

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

SUPERVISOR_SYSTEM_PROMPT = """
You are a Supervisor orchestrating a team of AI agents. 
Given the current task and conversation, decide who should act next.

Available agents:
- researcher: Gathers information from the web
- coder: Writes and executes Python code
- writer: Produces final written output
- FINISH: The task is complete

Respond ONLY with a JSON object: {{"next": "<agent_name>"}}
"""

class RouterOutput(BaseModel):
    next: str

def supervisor_node(state: AgentState) -> AgentState:
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        *state["messages"],
    ]
    response = llm.with_structured_output(RouterOutput).invoke(messages)
    return {"next_agent": response.next}
```

---

## Step 5: Define Agent Node Wrappers

Each agent node wraps the underlying agent and updates the shared state.

```python
def research_node(state: AgentState) -> AgentState:
    result = research_agent.invoke(state)
    return {
        "messages": result["messages"],
        "research_output": result["messages"][-1].content
    }

def code_node(state: AgentState) -> AgentState:
    result = code_agent.invoke(state)
    return {"messages": result["messages"]}

def writer_node(state: AgentState) -> AgentState:
    result = writer_agent.invoke(state)
    return {
        "messages": result["messages"],
        "final_output": result["messages"][-1].content
    }
```

---

## Step 6: Assemble the Graph

```python
from langgraph.graph import StateGraph, END

# Initialize graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", research_node)
workflow.add_node("coder", code_node)
workflow.add_node("writer", writer_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Conditional routing from supervisor
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "writer": "writer",
        "FINISH": END,
    }
)

# All agents report back to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")
workflow.add_edge("writer", "supervisor")

# Compile
graph = workflow.compile()
```

---

## Step 7: Run the Pipeline

```python
initial_state = {
    "messages": [
        {"role": "user", "content": (
            "Research the top 3 vector databases for RAG in 2024, "
            "write Python code to benchmark their query latency, "
            "then write a summary report of your findings."
        )}
    ],
    "task": "Vector DB research and benchmark",
    "next_agent": "",
    "research_output": "",
    "final_output": "",
}

# Stream the execution
for step in graph.stream(initial_state, {"recursion_limit": 20}):
    for node_name, output in step.items():
        print(f"\n{'='*50}")
        print(f"Node: {node_name}")
        if "messages" in output:
            print(output["messages"][-1].content[:500])
```

---

## Adding Memory with Checkpointing

For long-running or multi-turn workflows, add **persistence** with LangGraph's checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Each run is tied to a thread_id
config = {"configurable": {"thread_id": "session_001"}}

result = graph.invoke(initial_state, config=config)

# Resume from checkpoint later
follow_up = {
    "messages": [{"role": "user", "content": "Now compare pricing for those databases."}]
}
result2 = graph.invoke(follow_up, config=config)  # Remembers full prior context
```

---

## Human-in-the-Loop

LangGraph supports **interrupting** the graph for human review before critical steps:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["writer"]  # Pause before writer runs
)

# After review, resume
graph.invoke(None, config=config)  # Pass None to continue from checkpoint
```

---

## Production Patterns & Best Practices

### ✅ Design Patterns

| Pattern | When to Use |
|---|---|
| **Supervisor** | General task delegation across diverse agents |
| **Sequential Pipeline** | Fixed, ordered steps (ETL-style workflows) |
| **Parallel Fan-out** | Independent subtasks that can run concurrently |
| **Hierarchical** | Complex tasks needing sub-supervisors |

### ✅ Reliability

- **Set `recursion_limit`** to prevent infinite agent loops
- **Add validation nodes** between agents to catch bad outputs early
- **Use structured outputs** (`with_structured_output`) for routing decisions
- **Log every node transition** for debugging and auditability

### ✅ Cost Optimization

- Route simple subtasks to cheaper models (e.g., `gpt-4o-mini`)
- Cache tool results with `@lru_cache` or Redis for repeated queries
- Use `interrupt_before` to review expensive steps before execution

### ✅ Observability

```python
# Integrate LangSmith for full tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi-agent-prod"
```

---

## Real-World Use Case: Legal AI Pipeline

Here's how this architecture maps to a **legal document analysis system** (a pattern applicable to any domain):

```
User Query: "Summarize this employment contract and flag any non-standard clauses"
     │
     ▼
┌──────────────┐
│  Supervisor  │
└──────┬───────┘
       │
  ┌────┴──────────────┐
  ▼                   ▼
┌──────────┐    ┌──────────────┐
│  RAG     │    │   Clause     │
│  Agent   │    │  Classifier  │
│(retrieve │    │   Agent      │
│  docs)   │    │              │
└────┬─────┘    └──────┬───────┘
     │                 │
     └────────┬─────────┘
              ▼
       ┌────────────┐
       │  Writer    │
       │  Agent     │
       │ (summary + │
       │  flags)    │
       └────────────┘
```

Each agent focuses on what it does best — retrieval, classification, and generation — while the supervisor ensures the right agent is engaged at the right time.

---

## Conclusion

Multi-agent systems with LangChain and LangGraph unlock a new tier of LLM application complexity. The key principles to take away:

1. **Decompose** complex tasks into specialized agent roles
2. **Use shared state** for clean inter-agent communication
3. **The Supervisor pattern** scales well across most real-world use cases
4. **Checkpointing and HITL** are essential for production reliability
5. **Observe everything** — LangSmith traces save hours of debugging

The architecture described here is the same foundation powering production legal AI, financial analysis, and research automation systems being built today.

---

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Multi-Agent Concepts](https://python.langchain.com/docs/concepts/agents/)
- [LangSmith Observability](https://smith.langchain.com/)
- [CrewAI vs LangGraph: When to Use Which](https://blog.langchain.dev/)

---

*Have questions or want to see a deeper dive on any section? Drop a comment below or reach out on GitHub [@tuanquang95](https://github.com/tuanquang95).*
