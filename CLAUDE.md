# CLAUDE.md

Guidance for Claude Code when working in this repository. User-facing docs live in `README.md`.

## What this is

An agentic data pipeline. A LangGraph agent (Google Gemini via `langchain-google-genai`) scrapes 60 books from
books.toscrape.com (catalogue pages 1-3) by choosing tool calls itself. Data flows SQLite Bronze → Silver → Gold
in `data/books.db`. A Streamlit dashboard reads Gold and Silver.

## Commands

Always use the project venv (`.venv/`).

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest                                         # offline, ~1s, no API key needed
python -m src.run_pipeline                     # needs the provider's API key in .env; ~1-2 min
python -m src.run_pipeline --simulate-failure  # or SIMULATE_FAILURE=1
streamlit run dashboard/app.py
```

Run modules from the project root (`python -m src....`), not as files. `src` is imported as a package.

## Layout

- `src/config.py`: every setting (model, DB path, delays, User-Agents, `MAX_RETRIES_PER_URL`, `RECURSION_LIMIT`).
  Reads `.env`. Put new knobs here, never hard-coded in other modules.
- `src/db.py`: `SCHEMA` string plus helpers. `connect()` reads `config.DB_PATH` at call time, so tests can
  monkeypatch it.
- `src/tools.py`: pure parsers (`parse_price`, `parse_rating`, `parse_stock`, `parse_listing`, `parse_product`),
  `fetch()` (politeness delay and failure injection), the module-level `STATE` (`RunState`), and the 9 tools.
  `TOOLS` / `TOOLS_BY_NAME` sit at the bottom.
- `src/agent.py`: `SYSTEM_PROMPT`, `get_llm()` (Gemini, rate-limited for the free tier), `build_graph(llm=None)` (an LLM can be injected for offline testing),
  `run_agent()`.
- `src/run_pipeline.py`: CLI. `dashboard/app.py`: Streamlit app.
- `tests/test_pipeline.py` plus `tests/fixtures/*.html`: trimmed real pages. Tests never hit the network.

## Rules to keep

- **The LLM decides every step.** Don't add Python loops that scrape or save outside of agent tool calls.
- **Tools never raise** and always return `{"ok": bool, ...}`. New tools are decorated `@tool` then `@logged`
  (in that order, `@tool` outermost) and added to `TOOLS`.
- **Every tool call is logged** to the console and to `agent_logs` through `@logged` / `log_event()`. Special
  statuses: `FAILURE_INJECTED`, `RETRY`, `RECOVERED`, `GAVE_UP`, `ERROR`. The dashboard highlights them through
  `HIGHLIGHT` in `dashboard/app.py`, so keep both in sync.
- **Bronze stores raw strings** (`£51.77`, `star-rating Three`, `In stock (22 available)`). All parsing and casting
  happens in `save_clean_data` on the way to Silver.
- **Silver must stay idempotent:** `PRIMARY KEY(product_url)` plus `INSERT ... ON CONFLICT DO UPDATE`. Rerunning
  must leave exactly 60 rows. Gold tables are deleted and rebuilt from Silver on each save.
- **The summary may only use numbers from `get_facts`.** Keep that rule in `SYSTEM_PROMPT`.
- **Be polite:** keep tool execution sequential (custom `tool_node`, not the prebuilt parallel `ToolNode`),
  with a 0.3s delay, a timeout and a clear User-Agent.
- **No secrets in code.** The API key (`GOOGLE_API_KEY`) comes only from `.env` (git-ignored). Update `.env.example` when adding
  env vars.

## Testing changes without an API key

Don't call the real API from tests. To exercise the graph end to end, pass a fake chat model to
`run_agent(llm=...)`. It needs a `bind_tools()` that returns itself and a `_generate()` that returns `AIMessage`s
with `tool_calls`. Delete `data/books.db` afterwards if the run wrote placeholder data.

## Style

Simple, readable, commented. Small functions, type hints, f-strings, no extra frameworks. Match the existing
comment density.
