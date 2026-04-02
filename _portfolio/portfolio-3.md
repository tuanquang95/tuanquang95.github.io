---
title: "Multi-Agent Legal AI System"
excerpt: "Architected multi-agent legal AI system with LangGraph/LangChain for employment law"
collection: portfolio
---

# ⚖️ Legal AI — Multi-Agent System for Employment Law

[![Framework](https://img.shields.io/badge/Framework-LangGraph%20%2F%20LangChain-blue?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Database](https://img.shields.io/badge/Database-Neo4j%20%26%20FAISS-008CC1?style=flat-square&logo=neo4j)](https://neo4j.com/)
[![Cloud](https://img.shields.io/badge/Cloud-AWS%20Bedrock-FF9900?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/bedrock/)
[![Observability](https://img.shields.io/badge/Observability-LangSmith-DB7093?style=flat-square)](https://www.langchain.com/langsmith)
[![Pipeline](https://img.shields.io/badge/Pipeline-FastAPI%20%2B%20Airflow-017CE2?style=flat-square&logo=apache-airflow)](https://airflow.apache.org/)

> **Intelligent legal research and reasoning** — an enterprise-grade multi-agent system performing hybrid retrieval and entity relationship analysis across 100K+ employment law documents.

---

## 📌 Overview

Legal research requires high precision, exhaustive citation recall, and the ability to understand complex entity relationships (cases, statutes, parties). This project implements a **Multi-Agent Legal AI System** focused on employment law. 

The system leverages a **hybrid retrieval strategy** (combining keyword-based BM25 and semantic FAISS via Reciprocal Rank Fusion) to ensure relevant legal precedents are never missed. By integrating a **Neo4j knowledge graph**, the agents can reason about relationships between legal entities that traditional vector search might overlook.

---

## ⚙️ System Specifications

| Property | Value |
|---|---|
| **Domain** | Employment Law (USA) |
| **Document Corpus** | 100K+ Statutes, Case Laws, and Regulations |
| **Retrieval Strategy** | Hybrid (BM25 + FAISS) with Reciprocal Rank Fusion (RRF) |
| **Architecture** | Multi-Agent Directed Acyclic Graph (DAG) via LangGraph |
| **Monitoring** | LangSmith Traceability & Observability |
| **Deployment** | AWS Bedrock, ECR, ECS, Lambda |

### 🎯 Key Performance Indicators (KPIs)

- **Recall@K**: Evaluated to ensure exhaustive citation coverage.
- **MRR (Mean Reciprocal Rank)**: Benchmarked to ensure the most relevant precedents appear first.
- **Structural Integrity**: Pydantic models for strictly typed legal summaries.

---

## 🧠 Approach

### Retrieval & Ingestion Pipeline

```
Document Source (100K+)
      │
      ▼
 Airflow Orchestration
      │
      ├──► Chunking & Embedding (FAISS)
      └──► Entity Extraction (Neo4j Graph Store)
      │
      ▼
  Query Execution
      │
      ├──► BM25 (Keyword) ──┐
      │                     ├──► RRF (Reciprocal Rank Fusion)
      └──► FAISS (Semantic) ┘
               │
               ▼
      Multi-Agent Reasoning (LangGraph)
               │
               ▼
      Structured Output (Pydantic)
```

### Key Techniques

- **Hybrid Search (BM25 + FAISS)**: Chose this dual-approach after benchmarking showed an **18% improvement** in legal citation recall compared to pure vector search.
- **Graph-Augmented Generation**: Uses Neo4j to store and query entity relationships, allowing agents to identify "conflicts of interest" or "connecting precedents" across disparate cases.
- **Multi-Agent Orchestration**: LangGraph manages specialized agents (Researcher, Writer, Citator) with stateful memory and feedback loops.
- **LangSmith Observability**: Real-time evaluation and debugging of agent reasoning chains to minimize hallucinations.
- **Pydantic Validation**: Ensures every legal response contains structured citations, case references, and valid JSON payloads for downstream services.

---

## 📁 Repository Structure

```
legal-ai-agent/
├── airflow/                    # Ingestion DAGs and processing tasks
│   └── dags/
│       └── ingest_legal_docs.py
├── src/
│   ├── agents/                 # LangGraph agent definitions
│   │   ├── graph.py            # Main state machine
│   │   └── nodes/              # Researcher, Summarizer, Citator
│   ├── retrieval/              # Hybrid search logic (BM25, FAISS)
│   ├── ingestion/              # Document parsing and Neo4j loading
│   └── api/                    # FastAPI application layer
├── eval/                       # LangSmith evaluation scripts
│   └── tests/                  # Recall@K and MRR benchmarks
├── config/
│   └── pydantic_models.py      # Structured output schemas
├── docker/                     # ECR deployment configurations
└── README.md
```

---

## 🚀 Deployment & Infrastructure

The system is architected for enterprise scale on **AWS**:

1. **AWS Bedrock**: Serverless LLM orchestration (Claude 3.5 / Llama 3).
2. **Amazon ECR/ECS**: Containerized FastAPI endpoints for high availability.
3. **AWS Lambda**: Event-driven processing for incoming document updates.
4. **Apache Airflow**: Managed orchestration for periodic re-indexing of 100K+ legal PDF/MD files.

---

## 📊 Evaluation Results

| Strategy | Recall@10 | MRR | Note |
|---|---|---|---|
| Pure Vector (FAISS) | 0.72 | 0.65 | Misses specific legal terminology |
| Pure Keyword (BM25) | 0.68 | 0.60 | Misses semantic intent |
| **Hybrid (BM25 + FAISS + RRF)** | **0.85 (+18%)** | **0.78** | **Best for Legal Citations** |

> *Testing on a benchmark of 5,000 legal queries confirmed that RRF effectively blends the strengths of keyword matching for specific statutes with semantic matching for conceptual legal arguments.*

---

## 🔍 Challenges & Observations

- **Citation Precision**: Legal documents require exact citations. Using RRF was critical because law involves very specific terms (e.g., "FLSA Section 216(b)") that vector embeddings sometimes "smear" with similar concepts.
- **Graph Complexity**: Modeling 100K+ documents in Neo4j required significant schema optimization to keep relationship traversal under 200ms.
- **Hallucination Control**: Implementing self-correction loops in LangGraph reduced citation errors by ensuring a "Citator" agent verifies every reference against the retrieval context.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), LangChain |
| **LLMs** | AWS Bedrock (Claude 3.5 Sonnet) |
| **Vector DB** | FAISS |
| **Graph DB** | Neo4j |
| **API Framework** | FastAPI |
| **Observability** | LangSmith |
| **Data Orchestration** | Apache Airflow |
| **Cloud Infra** | AWS (ECS, Lambda, ECR, Cloudwatch) |

---

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Neo4j Vector Search & Graph RAG](https://neo4j.com/developer/graph-rag/)
- [Reciprocal Rank Fusion (RRF) Explained](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Pydantic Official Documentation](https://docs.pydantic.dev/)