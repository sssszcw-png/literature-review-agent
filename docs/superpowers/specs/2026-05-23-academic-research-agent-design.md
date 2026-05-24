# Academic Deep Research Agent — Design Spec

**Last updated**: 2026-05-24
**Status**: Implemented (131 tests passing, E2E pipeline working, Chinese translation support added)

## Overview

An AI agent that performs iterative deep research on academic topics. Input a research question, the agent searches academic databases, reads papers, builds a provenance-tracked knowledge map, and produces an IEEE-formatted literature review — all with controllable depth and cost.

---

## Architecture: Three-Round Iterative Research

### Round 1 — Broad Exploration
- LLM decomposes the research question into multiple search queries
- Concurrent search via Semantic Scholar API + arXiv API
- Read all returned abstracts, extract claims and relationships
- Build initial Knowledge Map (Pydantic models, JSON-serializable)
- Detect knowledge gaps via LLM

### Round 2 — Deep Dive (Looped)
- Gaps classified: `critical` / `important` / `nice_to_have`
- Targeted search for critical gaps
- Read full-text PDFs (via Docling) when abstract is insufficient; abstracts only in Round 1
- **Smart text selection**: Full-text Markdown from Docling is split by section headers; high-value sections (Methods, Experiments, Results, Discussion, Conclusion) are prioritized over low-value ones (Introduction, Related Work) within a fixed 8,000-character budget
- Update knowledge map and evaluate saturation via **LLM-as-a-Judge**:
  - Evaluator prompt takes (Gap + related KnowledgeNodes) → outputs three 0.0-1.0 scores:
    - `coverage`: how thoroughly do collected sources address this gap?
    - `source_quality`: weighted by venue prestige, citation count, recency
    - `consensus`: do multiple sources agree (high) or contradict (low)?
  - `saturation = avg(coverage, source_quality, consensus)` computed in Python
  - JSON parse failures (~1/12) gracefully degrade to `SaturationScores(0, 0, 0)`
- Loop until: all critical gaps have saturation >= 0.7, OR no improvement for 2 consecutive rounds
- Max 5 rounds total

### Round 3 — Report Generation
1. Generate review outline from knowledge map
2. Self-reflection: another LLM call critiques the outline (logic, coverage, flow)
3. Human-in-the-loop: display outline, user can approve / edit / abort; non-interactive mode defaults to approve
4. Write chapters in parallel (asyncio + semaphore-bounded concurrency)
5. Stitch chapters + consistency check (citation alignment, terminology, contradictions)
6. IEEE-formatted Markdown output with full provenance
7. **Chinese translation** (opt-in via `--zh` flag): full report translated via single LLM call, preserving IEEE citations, markdown structure, and untranslated References section; saved as `{slug}_zh.md` alongside the English report

---

## State Graph Topology

```
START
  │
plan_queries → search → read → update_knowledge_map → detect_gaps
  ↑                                                        │
  │                                    ┌───────────────────┤
  │ (critical gaps remain)            │ (evaluate needed)  │ (saturated or max rounds)
  │                                    ↓                    │
  └────────────────────── evaluate_saturation               │
                                   │                        │
                                   └──→ detect_gaps ────────┘
                                                            ↓
                                                    generate_outline
                                                            │
                                                 ┌──────────┤
                                                 │ HITL     │ (abort/quit)
                                                 ↓          ↓
                                           write_report    END
                                                 │
                                                END
```

**8 nodes**: `plan_queries`, `search`, `read`, `update_knowledge_map`, `detect_gaps`, `evaluate_saturation`, `generate_outline`, `write_report`

Conditional routing at `detect_gaps` and `generate_outline`.

When `--zh` flag is set, `write_report_node` appends a final translation step: the complete English report is sent to the LLM in a single call for full-document Chinese translation.

---

## Core Components

### 1. LangGraph State Machine
- 8 state nodes with conditional routing
- `MemorySaver` for in-memory state persistence during graph execution
- `CheckpointManager` saves JSON state snapshots to `checkpoints/` after each node transition for persistent crash recovery
- AgentState carries `output_zh: bool` and `final_report_zh: str` for Chinese translation support
- `--resume <thread_id>` loads last checkpoint and continues
- Human-in-the-loop at outline confirmation uses `asyncio.to_thread(input, ...)` for non-blocking I/O; in non-interactive contexts, defaults to approve

### 2. LLM Client
- DeepSeek API via OpenAI-compatible SDK (`AsyncOpenAI`)
- `LLMClient.complete()` for raw text; `LLMClient.complete_json()` for structured output
- `_extract_json()` handles markdown fences, trailing commas, and LaTeX escape sequences (`\cdot`, `\frac`, etc.)
- `_fix_json_escapes()` repairs invalid JSON backslash sequences from LLM outputs
- Prompt templates in `prompts/` directory (12 `.txt` files) loaded by `PromptManager`
- **Safe prompt rendering**: regex-based whitelist substitution replaces only known `{key}` placeholders; unknown braces (e.g. LaTeX, JSON examples) are left untouched; missing variables raise `KeyError`
- Retry: 3 attempts with exponential backoff (base 2.0s, max 60s)
- Translation: single-call full-document Chinese translation via `prompts/translate_to_chinese.txt`

### 3. Knowledge Map

```python
class PaperMeta(BaseModel):
    paper_id, title, authors, year, url, venue, citation_count, abstract,
    full_text_available, source

class Evidence(BaseModel):
    paper: PaperMeta
    section: str
    quote: str
    evidence_type: EvidenceType  # ORIGINAL_CLAIM | SUPPORTING | CONTRADICTING

class KnowledgeNode(BaseModel):
    id: str
    topic: str
    claim: str
    evidence: list[Evidence]
    confidence: float
    # Maintained by KnowledgeMap CRUD

class Gap(BaseModel):
    description: str
    severity: GapSeverity  # CRITICAL | IMPORTANT | NICE_TO_HAVE
    saturation: float = 0.0
    saturation_detail: dict = {}  # {coverage, source_quality, consensus}

class SaturationScores(BaseModel):
    coverage: float        # 0.0-1.0
    source_quality: float  # 0.0-1.0
    consensus: float       # 0.0-1.0
    # saturation property = avg of the three
```

Every claim in a KnowledgeNode must have at least one Evidence entry with paper, section, and verbatim quote — provenance is enforced.

### 4. Academic PDF Parser & Cache
- **Primary**: Docling (ML-driven, handles dual-column, formulas, tables) — installed as `docling>=2.0`
- **Fallback**: Marker (`marker-pdf>=1.0`) — graceful `ImportError` skip if not installed
- **Final fallback**: extract only abstract, mark `full_text_available=False`
- **Smart text selection** (`src/utils/text_selector.py`):
  - Splits parsed Markdown by `##` section headers
  - Skips: Abstract, References, Acknowledgments, Appendix
  - Low priority (fills budget remainder): Introduction, Related Work, Background
  - High priority (selected first): Methodology, Experiments, Results, Discussion, Conclusion (and 30+ recognized synonyms)
  - Budget: 8,000 characters to keep LLM input tokens bounded
- **Cache**: parsed text saved to `cache/papers/{paper_id}.json`; re-read on subsequent runs
- Model downloads: HuggingFace mirror (`HF_ENDPOINT=https://hf-mirror.com`) configured in `.env`

### 5. Search Engine
- **Semantic Scholar API**: title, abstract, authors, year, citation count, venue — rate-limited at configured interval
- **arXiv API**: supplementary full-text PDF access — rate-limited at 1 req/s
- `SearchEngine` orchestrates parallel queries via `bounded_gather(max_concurrent=5)`
- Deduplication by paper_id across sources

### 6. Query Planner
- Round 1 (broad): LLM generates 3-5 general queries from the research question
- Round 2+ (targeted): LLM generates queries targeting specific unsaturated gaps

### 7. Saturation Evaluator (LLM-as-a-Judge)
- Evaluator prompt takes gap description + related KnowledgeNodes with evidence
- Outputs `{coverage, source_quality, consensus, rationale}`
- JSON parse failures default to 0.0 for all scores; ~1/12 failure rate, benign impact (capped at 2 extra rounds)
- `consecutive_no_improvement` counter exits the loop after 2 rounds without progress

### 8. Report Writer
- **Outline generation** with self-reflection critique + HITL confirmation; user edits use dedicated `revise_outline` prompt
- **Parallel chapter writing**: `asyncio.gather` with `Semaphore(max_concurrent=5)`
- **Stitching**: `report/stitch.py` assembles chapters + reference list using shared `format_reference_entry()` from `formatter.py`
- **Consistency check**: cross-chapter citation alignment, terminology, contradiction detection via LLM; truncation controlled by `CONSISTENCY_CHECK_MAX_CHARS` constant
- **IEEE formatting**: citation rules in `src/config/constants.py`; formatting via `report/formatter.py`
- Citation index built once, shared across all chapters and reference list
- **Chinese translation** (opt-in): `--zh` flag triggers full-report translation via single LLM call after English report is saved; preserves IEEE citations, markdown structure, and keeps References section untranslated; saved as `{slug}_zh.md`; failure gracefully degrades with warning

---

## Project Structure

```
p2/
├── research.py              # CLI entry point (--zh flag for Chinese translation)
├── pyproject.toml           # Dependencies
├── .env                     # API keys, HF_ENDPOINT
├── src/
│   ├── agent/               # AgentState (with output_zh, final_report_zh), graph construction, routing
│   ├── cli/                 # ResearchCLI orchestrator, Rich progress display, display utilities
│   ├── config/              # Settings (Pydantic BaseSettings), constants (CONSISTENCY_CHECK_MAX_CHARS, etc.)
│   ├── knowledge/           # Pydantic models, KnowledgeMap CRUD, provenance validation
│   ├── llm/                 # LLMClient (OpenAI SDK → DeepSeek), PromptManager (regex safe rendering)
│   ├── nodes/               # 8 LangGraph node functions
│   ├── pdf/                 # PaperParser orchestrator, Docling/Marker wrappers, downloader, cache
│   ├── report/              # Chapters (parallel), stitch, formatter (shared format_reference_entry)
│   ├── search/              # SemanticScholarClient, ArxivClient, SearchEngine, search models
│   └── utils/               # RateLimiter, bounded_gather, text_selector, retry, dedup, logging, checkpoints
├── prompts/                 # 12 LLM prompt templates (.txt, incl. revise_outline, translate_to_chinese)
├── tests/                   # 131 tests (unit)
├── reports/                 # Output directory (English + optional Chinese _zh.md)
├── cache/                   # PDF and paper cache
└── checkpoints/             # Persistent checkpoint JSON files
```

## Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| Agent framework | LangGraph | State machine, retry, checkpoint, human-in-the-loop |
| LLM | DeepSeek API (`deepseek-chat`) | Cost-effective, OpenAI-compatible interface |
| PDF parsing | Docling (primary), Marker (fallback) | Academic paper layout handling |
| Text selection | Custom `select_text()` | Section-aware, prioritizes methods/results over intro |
| PDF cache | Local `cache/papers/` | Avoid re-parsing same paper across sessions |
| Knowledge graph | Pydantic models, JSON-serializable | Lightweight, sufficient for ~300 nodes |
| Academic search | Semantic Scholar API + arXiv API | Free, structured metadata |
| CLI | rich library | Progress bars, panels, Markdown rendering |
| Async | asyncio + aiohttp | Parallel search + parallel chapter writing + parallel PDF parsing |
| Citation | IEEE format via prompt rules | `[1]` inline, `## References` at end |

## CLI Interface

```
python research.py "<research question>" [--max-rounds 5] [--output-dir reports/]
                                         [--resume <thread_id>] [--no-checkpoint]
                                         [--verbose] [--list-checkpoints] [--cleanup <thread_id>]
                                         [--zh]
```

- `--zh`: Opt-in Chinese translation of the final report (saved as `{slug}_zh.md`)
- Research question validated for non-empty and <= 2000 characters
- Progress display per round: search results count, papers read, knowledge nodes, gaps detected, saturation scores. Outline displayed for user confirmation before final write.
- Chinese report preview shown after English preview when `--zh` is set

## Error Handling

- External services are unreliable — every call has retry + fallback
- PDF parsing failure does not block the pipeline (falls back to abstract)
- JSON parse failures from LLM gracefully degrade to defaults
- Chinese translation failure is caught and logged; English report is unaffected (translation runs after English save)
- Max round limit + no-improvement counter prevent infinite loops
- Checkpoints allow resumption after crash or interrupt
- Degraded results explicitly marked (`full_text_available=False`)
- Input validation rejects empty or excessively long (>2000 chars) research questions

## Known Limitations

- Semantic Scholar rate-limiting (429) → pipeline relies mostly on arXiv
- Docling model download (~500MB) on first run via HuggingFace
- Occasional JSON parse failures from LLM (~1/12 in saturation evals) — gracefully handled
- Some PDF downloads get 403 (paywalled papers) — abstract used as fallback
- Marker parser optional; not installed by default

## Non-Goals (v1)

- Web UI (CLI only; API interface designed for future web layer)
- Graph database (JSON in-memory, <500 nodes)
- Multi-user / concurrent research sessions
- Non-academic search sources
- Real-time streaming output
