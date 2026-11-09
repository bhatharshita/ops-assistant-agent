# Ops Assistant: ask your logs and GitHub issues in plain English

[![CI](https://github.com/<your-github-username>/ops-assistant-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-github-username>/ops-assistant-agent/actions/workflows/ci.yml)

A multi-agent assistant for on-call engineers. It answers questions like
*"Which node had the most OOMKilled containers?"* or *"How was the etcd
'database space exceeded' problem fixed?"* by combining **exact SQL counts over logs**,
**hybrid RAG over GitHub issues** and a **knowledge graph** that links issues, components and errors.
Every answer cites its sources.

Built with **LangGraph, FastAPI, PostgreSQL + pgvector, React + TypeScript, MCP and Docker**.
It runs end to end with **no API key** (a deterministic offline mode), and switches to
Anthropic, OpenAI or any OpenAI-compatible model (Mistral, Ollama, ...) when you add a key.

![Answer with cited sources](docs/screenshot-sources.jpg)

> **Demo GIF:** _placeholder: record the UI answering the example questions and save it as `docs/demo.gif`._

---

## The problem

When something breaks, the answer is usually spread over three places:

1. **Logs** tell you *what* is happening and *how often*, but nobody wants to write
   `GROUP BY` queries at 3 a.m.
2. **Issue trackers** tell you *why* it happens and *how others fixed it*, but keyword
   search misses issues phrased differently.
3. **Relationships**: which component an error comes from, and which other issues touch
   that component. These live in people's heads.

Plain RAG handles (2) reasonably, is bad at (1) (vector search can't count) and ignores (3).
This project routes each question to the right mix of tools.

## What it does

| Question | Route | What happens |
|---|---|---|
| Which node had the most OOMKilled containers? | `logs` | `log_stats(group_by="node", error_signature="OOMKilled")` → node-4, 24 lines |
| How was the etcd database space exceeded problem fixed? | `issues` | hybrid search → issue #1003 → "compact, defrag, disarm alarm, raise quota" [1] |
| Which issues are related to etcd? | `graph` | `graph_neighbors("etcd")` → #1003, #1009, #1017 + errors seen in etcd logs |
| What errors does the kubelet log most and how do I fix them? | `mixed` | log counts **and** issue fixes in one answer |
| What is the capital of France? | `out_of_scope` | polite refusal, no retrieval or tool calls |

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (python -m ops_assistant.ingest)"]
        GH[GitHub issues<br/>public API / sample] --> L[Loaders]
        LOGS[Log files<br/>sample / Loghub] --> L
        L --> CH[Chunking<br/>issues: paragraphs + header<br/>logs: digests + windows]
        CH --> EMB[Embeddings<br/>OpenAI-compatible or local hashing]
        L --> KG[Entity extraction<br/>components + error signatures]
    end

    subgraph PG["PostgreSQL 16 + pgvector"]
        CHUNKS[(chunks<br/>vector + tsvector)]
        LL[(log_lines)]
        GRAPH[(kg_nodes / kg_edges)]
    end

    EMB --> CHUNKS
    L --> LL
    KG --> GRAPH

    UI[React + TS UI<br/>chat, sources, trace] -- SSE --> API[FastAPI<br/>/chat/stream /chat /health]
    API --> AG[LangGraph agent]
    AG --> CHUNKS & LL & GRAPH
    MCP[MCP server<br/>stdio] --> CHUNKS & LL & GRAPH
    AG -. optional .-> LLM[(LLM API)]
```

### The agent graph

```mermaid
flowchart LR
    S((start)) --> R[router]
    R -- out_of_scope --> W[writer]
    R -- logs / issues / graph / mixed --> RT[retriever]
    RT --> A[analyst]
    A -- tool calls<br/>and budget left --> T[tools]
    T --> A
    A -- done --> W
    W --> E((end))
```

| Node | Job | With an LLM | Offline |
|---|---|---|---|
| **router** | classify the question and write a one-line plan | structured output (`RouteDecision` Pydantic model) | regex rules |
| **retriever** | hybrid RAG, filtered by route (logs → log chunks, issues → issue chunks) | same | same |
| **analyst** | decide which tools to call, with which arguments | `bind_tools` tool calling, can loop | rule-based planner that emits the same `AIMessage.tool_calls` |
| **tools** | run the calls, record results and extra sources | same | same |
| **writer** | grounded answer with `[n]` citations, streamed token by token | prompt with numbered sources + tool results | template over the same evidence |

The analyst ⇄ tools loop is capped by `MAX_TOOL_CALLS`. Every node pushes progress
events through LangGraph's custom stream (`get_stream_writer`), which FastAPI forwards as
server-sent events: `route → sources → tool_call → tool_result → token… → done`.
The UI's **Agent trace** tab shows them:

![Agent trace](docs/screenshot-trace.jpg)

### Tools

| Tool | Backed by | Use for |
|---|---|---|
| `search_knowledge_base` | hybrid pgvector + full-text search, RRF | "why", "how to fix", "has anyone seen" |
| `log_stats` | SQL `GROUP BY` over `log_lines` with whitelisted columns | "how many", "which node/component has the most", "when" |
| `graph_neighbors` | `kg_nodes` / `kg_edges`, 1-hop + 2-hop | "what's related to X" |
| `get_issue` | `documents` + `chunks` | full text of one issue |

The same functions are exposed to external clients through the **MCP server**
(`log_stats`, `graph_neighbors`, `search_knowledge_base`).

## Quick start (Docker)

```bash
cp .env.example .env        # optional: add ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up --build
```

- UI: http://localhost:3000
- API docs (Swagger): http://localhost:8000/docs
- Health: http://localhost:8000/health

Compose starts `db` (pgvector), runs the one-shot `ingest` job on the sample data, then
starts `api` and `ui` (nginx serving the built React app and proxying `/api` to FastAPI).

## Local development (without Docker for the app)

Requires Python 3.11+, Node 20+ and a PostgreSQL with pgvector
(easiest: `docker compose up -d db`).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp ../.env.example .env

python -m ops_assistant.ingest                 # sample data → Postgres
uvicorn ops_assistant.api.main:app --reload    # http://localhost:8000

cd ../frontend && npm install && npm run dev   # http://localhost:5173 (proxies /api)
```

`make help` lists shortcuts for all of these.

### Use real public data instead of the sample

```bash
cd backend
python scripts/fetch_github_issues.py --repo kubernetes/kubernetes --label kind/bug --limit 200 \
    --out ../data/raw/k8s_issues.json          # set GITHUB_TOKEN for higher rate limits
python scripts/fetch_loghub.py --dataset OpenStack   # public Loghub sample → ../data/raw/loghub
python -m ops_assistant.ingest --issues ../data/raw/k8s_issues.json --logs ../data/raw/loghub --reset
```

Any repository works (`--repo owner/name`). Unstructured log formats still load: the
level is detected and the component defaults to the file name.

### MCP server

```bash
cd backend && python -m ops_assistant.mcp_server      # stdio transport
python scripts/mcp_client_demo.py                     # launches it and calls log_stats
```

To use it from an MCP client such as Claude Desktop:

```json
{"mcpServers": {"ops-assistant": {
  "command": "/absolute/path/to/backend/.venv/bin/python",
  "args": ["-m", "ops_assistant.mcp_server"],
  "env": {"DATABASE_URL": "postgresql://ops:ops@localhost:5432/ops"}}}}
```

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)), loaded by
[`config.py`](backend/ops_assistant/config.py).

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto` → Anthropic if `ANTHROPIC_API_KEY`, else OpenAI if `OPENAI_API_KEY`, else `offline` |
| `OPENAI_BASE_URL` | – | any OpenAI-compatible API (Mistral, Ollama, vLLM, gateways) |
| `EMBEDDING_PROVIDER` | `auto` | `openai` (any OpenAI-compatible `/embeddings`) or `hashing` (local, no download) |
| `EMBEDDING_DIM` | `512` | must match the model; re-ingest with `--reset` after changing |
| `RETRIEVAL_MODE` | `hybrid` | `dense` or `hybrid` |
| `MAX_TOOL_CALLS` | `4` | budget for the analyst ⇄ tools loop |

## Evaluation

`backend/evals/dataset.jsonl` holds **24 labelled questions** (7 logs, 11 issues, 3 graph,
2 mixed, 1 out-of-scope). Each has the expected route, the relevant document ids, facts the
answer must contain and a reference answer.

```bash
cd backend
python -m evals.run_eval                        # current settings
python -m evals.run_eval --retrieval dense      # baseline
python -m evals.run_eval --out evals/results/my-run.json
```

| Metric | Meaning |
|---|---|
| route accuracy | router chose the labelled route |
| hit@5 / MRR | a relevant document is in the retriever's top 5 / mean of 1 ÷ rank of the first relevant hit |
| keyword recall | share of required facts (numbers, flags, node names) present in the answer |
| judge correctness / faithfulness | LLM-as-judge, 1–5, against the reference answer and the cited sources (only runs when an LLM is configured) |
| latency p50 / p95 | full agent run, wall clock |

### Results

Measured on the bundled sample data, offline mode (`hashing` embeddings, rule-based
agent, no LLM), PostgreSQL 16 + pgvector 0.6. Full per-question output is in
[`backend/evals/results/`](backend/evals/results/).

| Run | Retrieval | hit@5 | MRR | Route acc. | Keyword recall | p50 / p95 latency |
|---|---|---|---|---|---|---|
| 1. baseline | dense (pgvector only) | 0.870 | 0.719 | 0.875 | 0.948 | 1.1 / 2.2 ms |
| 2. baseline | hybrid (dense + full-text, RRF) | 0.913 | 0.703 | 0.875 | 0.948 | 1.7 / 2.6 ms |
| 3. **improved** | hybrid + **length-normalised** full-text rank | **0.913** | **0.757** | 0.875 | 0.948 | 1.6 / 2.4 ms |

**What changed between run 2 and 3.** The keyword half of hybrid search ranked with
`ts_rank_cd(tsv, query)`, which rewards *how many times* query words occur. Long log
windows repeat words like "pod" and "node" dozens of times, so they outranked the short issue
chunk that actually answered the question. Passing normalisation flag `2` divides the score by
the chunk length, the same idea as BM25's length normalisation. Keyword-only MRR went from
0.557 to 0.848, and hybrid MRR from 0.703 to 0.757 (commit `Length-normalise full-text
ranking`). I also tried capping chunks per document in the top-k; it made no measurable
difference, so I dropped it.

**Reading these numbers honestly.**
- 24 questions is a small, hand-written dev set, so treat differences of one question (~0.04) as noise.
- The HNSW index is approximate, so a fresh ingest can shift dense ranks slightly
  (a clean re-run of run 3 gave MRR 0.761 instead of 0.757).
- The offline router rules and this dataset were written by the same person, so offline
  route accuracy is optimistic. The interesting comparison is offline vs an LLM router on the
  same set.
- Latency is tiny because offline mode makes no LLM calls; with an LLM, expect seconds,
  dominated by generation.
- LLM-as-judge scores are not in the table because these runs had no API key. Add a key and
  re-run `python -m evals.run_eval` to get `judge_correctness` / `judge_faithfulness`.

## Design decisions and trade-offs

- **Postgres for everything (vectors, full text, logs, graph).** One database to run, back up
  and join across. pgvector's HNSW index is plenty for this scale. The trade-off: no
  Cypher and no graph algorithms. If the graph grew to millions of edges or needed
  multi-hop path queries, Neo4j (or Apache AGE inside Postgres) would be the next step.
- **Hybrid retrieval with Reciprocal Rank Fusion.** Ops text is full of exact tokens
  (`CrashLoopBackOff`, `--quota-backend-bytes`, `node-4`) that embeddings blur and keyword
  search nails. RRF fuses ranks, so the two very different score scales never need calibrating.
- **Structured tools instead of free-form Text-to-SQL.** `log_stats` takes typed arguments
  with a whitelisted `group_by`, and every value is a bound parameter. The LLM can't write
  arbitrary SQL, so injection and runaway queries are impossible by construction. The cost is
  flexibility: a new kind of question means adding a parameter or a tool. A guarded Text-to-SQL
  tool (read-only role, `SELECT`-only parser check, row limits) would be the next step.
- **Log digests as retrieval units.** Raw log lines are poor RAG chunks. One summary chunk
  per (component, error) pair ("38 ERROR lines with OOMKilled, mostly node-4…") lets semantic
  search answer "what's wrong with the kubelet?" while exact numbers come from SQL.
- **Rule-based entity extraction for the graph.** Regex dictionaries for components and error
  signatures are fast, free, deterministic and unit-testable. They miss entities that are not in
  the dictionary. LLM extraction at ingest time would raise recall at a per-document cost.
- **Offline mode as a first-class path.** The rule-based nodes honour the same contracts as
  the LLM nodes (the analyst returns real `tool_calls`), so CI and tests exercise the actual
  graph, tools, database and API without keys or flaky network calls.
- **SSE over WebSockets.** Answers stream one way, server to client. SSE is plain HTTP and works
  through proxies (nginx `proxy_buffering off`). The client reads the `fetch` body stream
  because `EventSource` can't POST.
- **Pluggable providers via OpenAI-compatible APIs.** One embeddings client covers OpenAI,
  Mistral, Azure-style gateways and Ollama by changing `base_url`. `EMBEDDING_DIM` is explicit
  because the pgvector column has a fixed size.

## Project structure

```
backend/
  ops_assistant/
    config.py            settings from env
    store.py             all SQL: pgvector, full-text, log stats, graph
    embeddings.py        OpenAI-compatible + local hashing embedders
    retrieval.py         dense / hybrid retriever, reciprocal rank fusion
    tools.py             agent tools (also used by the MCP server)
    llm.py               chat-model factory (Anthropic / OpenAI-compatible / offline)
    agent/
      graph.py           LangGraph nodes and edges
      state.py           shared state + reducers
      prompts.py         prompt templates
      offline.py         deterministic stand-ins for the LLM steps
    ingest/              loaders, chunking, entity extraction, graph building, CLI
    api/main.py          FastAPI app (SSE streaming)
    mcp_server.py        MCP server (stdio)
    schema.sql           tables + HNSW and GIN indexes
  evals/                 labelled dataset, harness, recorded results
  scripts/               fetch GitHub issues / Loghub logs, regenerate sample logs, MCP demo
  tests/                 pytest: unit tests + pgvector integration tests
frontend/                React + TypeScript (Vite), nginx config for Docker
data/sample/             synthetic issues + generated logs (see below)
docker-compose.yml       db + ingest + api + ui
.github/workflows/ci.yml lint, tests with a pgvector service, eval run, UI build, compose smoke test
```

## Testing

```bash
cd backend
pytest                                    # unit tests (no database needed)
TEST_DATABASE_URL=postgresql://ops:ops@localhost:5432/ops_test pytest   # + integration tests
cd ../frontend && npm test
```

Integration tests ingest the sample data into a **disposable** database (they drop the
tables), then check counts, search, the graph, the full agent and an MCP tool call. CI runs
everything against a `pgvector/pgvector:pg16` service container, then brings the full Docker
Compose stack up and calls the API through the nginx proxy.

## About the data

`data/sample/` is **synthetic**. The 24 issues were written for this project and modelled on
publicly documented Kubernetes failure modes (CrashLoopBackOff, etcd quota, CoreDNS timeouts…).
They are not copies of real GitHub issues. The logs are generated by
`backend/scripts/generate_sample_logs.py` with a fixed seed, so eval answers are stable. To
work with real public data, use the fetch scripts above. Fetched data goes to `data/raw/`,
which is git-ignored.

## Limitations and next steps

- Run the eval with a real LLM and embedding model, and add those rows to the table.
- Add a guarded Text-to-SQL tool for log questions the fixed `log_stats` arguments can't express.
- Incremental ingestion (only new issues / log lines) and scheduled re-sync from GitHub.
- Conversation memory (LangGraph checkpointer) for follow-up questions.
- Tracing and cost tracking (OpenTelemetry or LangSmith) per node and tool.

## License

MIT
