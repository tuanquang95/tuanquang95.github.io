---
title: "AI Health Counselor"
excerpt: "RAG-powered health counselor with medical safety guardrails"
collection: portfolio
---

# 🏥 AI Health Counselor — RAG-Powered Medical Guidance

[![Framework](https://img.shields.io/badge/Framework-LangChain-blue?style=flat-square&logo=langchain)](https://github.com/langchain-ai/langchain)
[![Database](https://img.shields.io/badge/Database-ChromaDB-008CC1?style=flat-square)](https://www.trychroma.com/)
[![Safety](https://img.shields.io/badge/Safety-Guardrails%20%2B%20Disclaimers-red?style=flat-square)](https://github.com/guardrails-ai/guardrails)
[![Memory](https://img.shields.io/badge/Memory-Sliding--Window-success?style=flat-square)](https://python.langchain.com/docs/modules/memory/)

> **Responsible AI for health counseling** — a retrieval-augmented generation (RAG) system providing structured, safety-first medical guidance over curated datasets with context-aware dialogue.

---

## 📌 Overview

Healthcare AI requires more than just high accuracy; it requires rigorous **safety guardrails**, context-aware memory, and the ability to strictly adhere to topic boundaries. This project implements a **RAG-powered Health Counselor** designed to provide informative, responsible medical guidance while minimizing hallucinations and ensuring mandatory disclaimers are always present.

The system utilizes **ChromaDB** for efficient retrieval over thousands of curated medical Q&A entries and **Pydantic** for strictly structured outputs to ensure consistency in medical advice formatting.

---

## ⚙️ Project Details

| Property | Value |
|---|---|
| **Domain** | Healthcare / Medical Q&A |
| **Data Source** | Curated Medical Q&A Datasets |
| **Retrieval Engine** | ChromaDB (Vector Search) |
| **Memory Strategy** | Sliding-Window Conversation Memory |
| **Routing** | Function Calling for Symptom Follow-up |
| **Safety** | Hallucination Detection & Topic Filtering |

### 🛡️ Safety & Responsibility Layer

- **Topic Boundary Filtering**: Automatically rejects queries outside the medical/health domain to prevent misuse.
- **Hallucination Detection**: Cross-references generated responses against retrieved context to flag potential factual errors.
- **Mandatory Disclaimers**: Automatically appends medical disclaimers to every response, emphasizing that the AI is not a substitute for professional medical advice.

---

## 🧠 Approach

### Dialogue & Reasoning Pipeline

```
User Query
      │
      ▼
 Safety Filter (Input Guardrails)
      │
      ▼
 Retrieval Step (ChromaDB)
      │
      ├──► Search curated medical Q&A
      │
      ▼
 Context-Aware Dialogue Engine (LangChain)
      │
      ├──► Sliding-window Memory (History)
      └──► Function Calling (Symptom Routing)
      │
      ▼
 Hallucination Detection (Verification)
      │
      ▼
 Structured Response (Pydantic + Disclaimer)
```

### Key Techniques

- **Context-Aware Multi-turn Dialogue**: Implemented stateful conversation management using sliding-window memory, significantly improving response coherence across long user sessions.
- **Symptom Follow-up Routing**: Leveraged **LLM function calling** to identify when a user mentions specific symptoms, triggering prioritized routing to specialized follow-up diagnostic questions.
- **RAG over Curated Data**: Eschewed broad web-searches for retrieval over high-quality, verified medical Q&A pairs to ensure grounding in expert knowledge.
- **Structured Output (Pydantic)**: Every response follows a rigid schema: `advice`, `supporting_evidence`, `suggested_next_steps`, and `disclaimer`.

---

## 📁 Repository Structure

```
health-counselor-ai/
├── data/
│   ├── curated_qa/             # Verified medical datasets
│   └── vectordb/               # ChromaDB index persistence
├── src/
│   ├── engine/                 # Core LangChain RAG logic
│   │   ├── chains.py           # Dialogue and retrieval chains
│   │   └── memory.py           # Sliding-window implementation
│   ├── safety/                 # Guardrails and filters
│   │   ├── hallucination.py    # Fact-checking against context
│   │   └── topic_filter.py     # Domain boundary enforcement
│   ├── routing/                # Function calling for symptom routing
│   └── schemas/                # Pydantic models for responses
├── notebooks/
│   └── EDA_and_Indexing.ipynb  # Data processing and embedding
├── requirements.txt
└── README.md
```

---

## 📊 Performance & Coherence

- **Response Coherence**: High stability maintained across 10+ turn conversations due to the sliding-window memory strategy.
- **Safety Compliance**: 100% adherence to topic filtering for non-medical queries during testing.
- **Retrieval Precision**: High overlap between generated advice and retrieved reference documents.

> *The system was built with a "Safety First" philosophy, prioritizing response grounding and disclaimer inclusion over conversational creativity.*

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| **Orchestration** | [LangChain](https://github.com/langchain-ai/langchain) |
| **Vector Store** | ChromaDB |
| **Validation** | Pydantic |
| **Logic** | Python |
| **LLMs** | OpenAI GPT-4o / Claude 3 Sonnet |
| **Metrics** | RAGAS (for retrieval/generation evaluation) |

---

## 📚 References

- [LangChain Documentation — Memory](https://python.langchain.com/docs/modules/memory/)
- [ChromaDB: The AI-Native Open Source Embedding Database](https://www.trychroma.com/)
- [Guardrails AI — Input/Output Validation](https://github.com/guardrails-ai/guardrails)
- [Pydantic Official Site](https://docs.pydantic.dev/)

---

*Focusing on responsible, grounded AI solutions for healthcare guidance.*
