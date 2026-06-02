# Agentic Research Assistant

An ongoing agentic AI project for researching complex questions with structured planning, evidence retrieval, and source-grounded answer synthesis.

The goal is to build a research assistant that can break a question into focused subtasks, choose the appropriate retrieval path for each task, gather evidence from public and internal sources, and produce a clear answer with traceable sources. The project is also a practical learning environment for agent orchestration, retrieval quality, memory design, evaluation, and production backend patterns.

## Current Architecture

```text
User Query
-> LangGraph Workflow
-> Planner Agent (Ollama / Qwen)
-> Researcher Agent
   -> Tavily Web Search
   -> Internal RAG Search
-> Evidence Synthesis (Gemini)
-> Final Response
```

## Project Status

### Completed

- Defined Pydantic schemas for research plans, subtasks, evidence, and final responses.
- Built a linear LangGraph workflow for planning, research, and synthesis.
- Added an LLM-powered planner that decomposes questions and routes subtasks to web or RAG retrieval.
- Added a researcher agent with Tavily web search and internal document retrieval.
- Implemented PDF text and table extraction with `pdfplumber`.
- Implemented multiple chunking strategies with source metadata and neighbor links.
- Added BGE embeddings with Sentence Transformers.
- Added lazy FAISS vector indexing and internal RAG retrieval.
- Added tests for chunking, embeddings, vector search, agent routing, and workflow behavior.

### In Progress

- Validate retrieval quality with representative research questions.
- Enrich citation metadata returned from internal retrieval.
- Add BM25 keyword retrieval and hybrid ranking.
- Add cross-encoder reranking for stronger result ordering.

### Planned

- Add episodic memory with reusable research summaries.
- Add LangGraph branching, fallback handling, and retry paths for weak evidence or tool failures.
- Build an evaluation dataset and LLM judge workflow.
- Expose the assistant through FastAPI endpoints.
- Add cache services for repeated queries and reusable retrieval results.
- Package the backend for Docker deployment.

## Planned Final Architecture

```text
FastAPI
-> Cache Lookup
-> LangGraph
   -> Load Relevant Episodic Memory
   -> Planner Agent
   -> Researcher Agent
      -> Tavily Web Search
      -> Hybrid RAG: FAISS + BM25
      -> Cross-Encoder Reranker
   -> Evidence Synthesis + Citations
   -> Write Episodic Memory
   -> Evaluation / Retry Logic
-> Response
```

## Retrieval Pipeline

The internal document path currently follows:

```text
PDF Documents
-> Text and Table Extraction
-> Metadata-Preserving Chunking
-> BGE Embeddings
-> FAISS Vector Index
-> Similarity Search
-> Evidence Items
```

The next retrieval milestone is hybrid search: combining semantic FAISS results with BM25 keyword matches, followed by cross-encoder reranking.

## Development Order

1. Validate current FAISS retrieval quality.
2. Enrich citation metadata.
3. Add BM25 retrieval and hybrid ranking.
4. Add cross-encoder reranking.
5. Add episodic memory.
6. Add conditional LangGraph edges, fallbacks, and retries.
7. Build evaluation workflows.
8. Add FastAPI endpoints and caching.
9. Package the backend with Docker.

## Local Configuration

Create a local `.env` file with the required API keys:

```text
TAVILY_API_KEY=...
GOOGLE_API_KEY=...
```

The planner currently uses a local Ollama model, so Ollama must also be available on the development machine.

Local documents and generated retrieval artifacts belong in `data/`. This directory is intentionally excluded from Git so PDFs, chunks, and embeddings are not committed accidentally.

## Repository Notes

- This project is under active development.
- Setup commands will be added after the dependency manifest is finalized.
- Do not commit `.env`, local documents, generated embeddings, or cache files.
