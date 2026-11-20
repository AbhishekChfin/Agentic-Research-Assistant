# Research Agent

A multi-agent research assistant that plans, researches, synthesizes, and evaluates answers using LLMs — inspired by [DeepLearning.AI's Agentic AI course](https://github.com/https-deeplearning-ai/agentic-ai-public).

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│   Planner    │  → Breaks query into subtasks (web / rag)
└──────┬───────┘
       ▼
┌──────────────┐
│  Researcher  │  → Executes each subtask via Tavily or FAISS RAG
└──────┬───────┘
       ▼
┌──────────────┐
│  Summarizer  │  → Synthesizes evidence into a final answer
└──────┬───────┘
       ▼
┌──────────────┐
│    Judge     │  → Evaluates answer quality; retries if needed
└──────────────┘
```

## Project Structure

```
.
├── main.py                 # FastAPI entry point (routes, background tasks)
├── src/
│   ├── planning_agent.py   # Planner, validator, workflow orchestrator
│   ├── agents.py           # Researcher, summarizer, judge agents
│   └── research_tools.py   # Tavily web search + FAISS RAG search
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css           # Styles
├── data/                   # PDF docs and processed embeddings
├── docker/
│   └── entrypoint.sh       # Container startup (Postgres + Uvicorn)
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Local Development

```bash
# 1. Clone and enter the project
cd research_assistant

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run the app
uvicorn main:app --reload
```

Open http://localhost:8000 to use the web UI.

### Docker

```bash
# Build
docker build -t research-agent .

# Run
docker run --rm -it -p 8000:8000 --env-file .env research-agent
```

## Environment Variables

| Variable         | Description              |
| ---------------- | ------------------------ |
| `GOOGLE_API_KEY` | Google Gemini API key    |
| `TAVILY_API_KEY` | Tavily search API key    |

## Tech Stack

- **FastAPI** + **Uvicorn** — Web framework and ASGI server
- **Gemini 2.5 Flash** — Planning, summarization, and evaluation
- **Tavily** — Real-time web search
- **FAISS** + **BGE embeddings** — Local RAG over PDF documents
- **Pydantic** — Structured I/O for all agents
- **Docker** + **PostgreSQL** — Containerized deployment with state persistence
