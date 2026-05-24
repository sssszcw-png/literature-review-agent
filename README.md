# Academic Deep Research Agent

An AI agent for **iterative academic literature review** built with [LangGraph](https://github.com/langchain-ai/langgraph) and any OpenAI-compatible LLM (DeepSeek, OpenAI, Groq, Ollama, etc.). It autonomously searches, reads, maps knowledge, identifies gaps, and generates a structured survey report — with provenance tracking and human-in-the-loop outline approval.

## Architecture

```
START
  │
plan_queries ──→ search ──→ read ──→ update_knowledge_map ──→ detect_gaps
      ↑                                                              │
      │                                          ┌───────────────────┤
      │                                          │ gaps saturated or  │
      │         critical gaps remain             │ max rounds         │
      │                                          ↓                    │
      └────────────────────────────── evaluate_saturation             │
                                                         │            │
                                                         └──→ detect_gaps
                                                                      ↓
                                                              generate_outline
                                                                      │
                                                           ┌──────────┤
                                                           │ HITL     │ abort
                                                           ↓          ↓
                                                     write_report    END
```

- **plan_queries** — Generates search queries from the research question and identified gaps
- **search** — Queries arXiv and Semantic Scholar in parallel
- **read** — Downloads and parses PDFs (via docling/marker), falls back to abstracts
- **update_knowledge_map** — Extracts claims and builds a provenance-tracked knowledge graph
- **detect_gaps** — Identifies gaps in the current knowledge coverage
- **evaluate_saturation** — Assesses whether additional rounds would yield diminishing returns
- **generate_outline** — Produces a structured report outline with HITL approval
- **write_report** — Generates the final survey report and optional Chinese translation

## Quick Start

### Prerequisites

- Python >= 3.11
- An API key from any OpenAI-compatible provider

### Installation

```bash
git clone https://github.com/sssszcw-png/literature-review-agent.git
cd literature-review-agent
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set your LLM_API_KEY (and LLM_BASE_URL / LLM_MODEL if not using DeepSeek)
```

### Usage

```bash
# Basic research
python research.py "What are the latest advances in retrieval-augmented generation?"

# With custom settings
python research.py "diffusion models for molecular generation" --max-rounds 8 --output-dir my_reports/

# Resume from a checkpoint
python research.py "..." --resume <thread_id>

# Generate Chinese translation
python research.py "large language model alignment techniques" --zh

# Manage checkpoints
python research.py --list-checkpoints
python research.py --cleanup <thread_id>
```

## Key Features

- **Multi-model support** — Works with any OpenAI-compatible API (DeepSeek, OpenAI, Groq, Ollama, etc.)
- **Multi-source search** — Queries arXiv and Semantic Scholar simultaneously
- **PDF parsing** — Full-text extraction via docling/marker with automatic fallback to abstracts
- **Knowledge mapping** — Structured claim extraction with provenance (paper → claim)
- **Gap detection** — LLM identifies uncovered aspects and knowledge gaps
- **Saturation evaluation** — Stops iterating when diminishing returns are detected
- **HITL outline review** — Approve or revise the report outline before writing
- **Checkpoint & resume** — Persists agent state after each node for crash recovery
- **Chinese output** — Optional Chinese translation of the final report
- **Rich CLI** — Progress bars, live status, and report preview in the terminal

## Project Structure

```
├── research.py              # CLI entry point
├── src/
│   ├── agent/               # LangGraph state machine & routing
│   ├── cli/                 # Rich-based CLI display and progress
│   ├── config/              # Pydantic settings (env-driven)
│   ├── knowledge/           # Knowledge graph models, map, provenance
│   ├── llm/                 # OpenAI-compatible LLM client & prompt helpers
│   ├── nodes/               # 8 graph nodes (one per pipeline step)
│   ├── pdf/                 # PDF downloader, parser, cache
│   ├── report/              # Outline, chapter generation, stitching, translation
│   ├── search/              # arXiv & Semantic Scholar clients
│   └── utils/               # Logging, retry, dedup, checkpoints, async
├── prompts/                 # LLM prompt templates (English → Chinese)
├── tests/                   # Unit and integration tests
└── docs/                    # Design specs
```

## Configuration Reference

All settings are loaded from environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | (required) | API key for your LLM provider |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `LLM_BASE_URL` | `https://api.deepseek.com` | Provider API endpoint |
| `LLM_TEMPERATURE` | `0.1` | LLM temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per response |
| `LLM_REQUEST_TIMEOUT` | `120` | Request timeout (seconds) |
| `DEFAULT_MAX_ROUNDS` | `5` | Research iteration rounds |
| `SATURATION_THRESHOLD` | `0.7` | Knowledge saturation cutoff |
| `MAX_CONCURRENT_SEARCHES` | `5` | Parallel search limit |
| `MAX_CONCURRENT_CHAPTERS` | `5` | Parallel chapter generation |
| `RETRY_MAX_ATTEMPTS` | `3` | Retry attempts |
| `SEARCH_MAX_RESULTS_PER_QUERY` | `20` | Results per search query |
| `PDF_MAX_SIZE_MB` | `50` | Max PDF file size |

See [.env.example](.env.example) for the full list.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
