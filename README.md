# iai-books-agent

A small **agentic** data pipeline: a LangGraph + Google Gemini agent scrapes the first 3 catalogue pages (60 books)
of [books.toscrape.com](https://books.toscrape.com/), stores them in a SQLite **Bronze → Silver → Gold** model,
writes a fact-grounded summary, and a Streamlit dashboard shows the results and the agent's decisions.

The LLM decides every step: it chooses which tool to call next, when to check status, which retry strategy to use,
and when it is done. There is no hard-coded scraping loop.

## Setup

```bash
cd iai-books-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env and set GOOGLE_API_KEY
```

`.env` settings:

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GOOGLE_MODEL` | `gemini-3.5-flash` | Gemini model used by the agent |
| `GOOGLE_FALLBACK_MODELS` | `gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite` | Tried in order if a model fails (quota used up, timeout) |
| `GOOGLE_REQUESTS_PER_MINUTE` | `4` | Client-side pacing; the Gemini free tier allows 5 requests/min, so a run takes ~4-5 min |
| `SIMULATE_FAILURE` | `0` | `1` = fail the first request to page 2 once |

## How to run

```bash
python -m src.run_pipeline                     # full run, prints each agent decision + tool result
python -m src.run_pipeline --simulate-failure  # injects one HTTP 503 on page 2; watch the agent recover
streamlit run dashboard/app.py                 # dashboard (has its own Run Pipeline button + checkbox)
pytest                                         # offline tests (saved HTML snippets, no network)
```

A run takes about 1–2 minutes, mostly because of the polite 0.3 s delay between the 63 requests.
Output lands in `data/books.db`.

What `--simulate-failure` looks like in the console (and in `agent_logs`):

```
fetch                FAILURE_INJECTED Simulated HTTP 503 for .../page-2.html
scrape_listing_page  ERROR            ... "error": "HTTP 503 Service Unavailable (simulated)"
get_status           OK               ... "failed_urls": {".../page-2.html": "HTTP 503 ..."}
retry_failed         RETRY            attempt 1/3 strategy=backoff url=.../page-2.html
retry_failed         RECOVERED        .../page-2.html recovered on attempt 1 using backoff
```

## Hosting on Streamlit Community Cloud

1. Push this folder to a GitHub repository. `.env` and `.streamlit/secrets.toml` are git-ignored, so your key
   is not pushed.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app** and pick the repo and branch.
   Set **Main file path** to `dashboard/app.py`.
3. Open **Advanced settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example`
   with your real `GOOGLE_API_KEY`.
4. Deploy. Streamlit installs `requirements.txt` and serves the dashboard.

Things to know when hosted:
- **Shipped snapshot:** `data/books.db` is committed, so the dashboard shows data before anyone clicks Run.
- **Temporary storage:** the cloud filesystem resets when the app restarts or redeploys, so the dashboard
  goes back to the committed snapshot.
- **One run at a time:** if two visitors click **Run Pipeline**, the second gets "already in progress".
- **Shared quota:** the Gemini free tier allows about 5 requests/min and 20 requests/day *per model*. A run needs
  about 7–10 requests, so one model covers about 2 runs a day. The fallback chain (4 models) raises that to about 8.
  Anyone with the URL can click Run, so keep the app private (Streamlit lets you restrict viewers) if
  that matters to you.

## Architecture

```
                 ┌─────────────────────── LangGraph ───────────────────────┐
  run_pipeline / │                                                          │
  dashboard ───▶ │  START ─▶ agent (Gemini + system prompt) ──tool calls──▶ tools node
                 │              ▲            │ no tool calls                 │ (runs tools one by one,
                 │              │            ▼                               │  logs each call)
                 │              └──────── END ◀── finish() called ──────────┘
                 └──────────────────────────────────────────────────────────┘
                                         │ tools
        scrape_listing_page / scrape_product_page / retry_failed ──▶ books.toscrape.com
                                         │
                                         ▼
   SQLite data/books.db
   bronze_books (raw strings, append-only) ──save_clean_data──▶ silver_books (typed, PK product_url, upsert)
                                                                   │
                                                                   ▼
          gold_kpis · gold_books_by_category · gold_rating_distribution · gold_top10_expensive · gold_summary
                                                                   │
   agent_logs (every tool call + FAILURE_INJECTED / RETRY / RECOVERED) ──▶ Streamlit dashboard
```

| File | Role |
|---|---|
| `src/config.py` | Settings from `.env`: model, paths, delays, retry and recursion limits |
| `src/db.py` | Schema and small SQLite helpers |
| `src/tools.py` | Parsers, polite HTTP fetch, failure injection, the 9 `@tool`s, logging |
| `src/agent.py` | System prompt, LangGraph graph, `run_agent()` |
| `src/run_pipeline.py` | CLI entrypoint |
| `dashboard/app.py` | Streamlit dashboard |
| `tests/test_pipeline.py` | Parsing and idempotency tests using `tests/fixtures/*.html` |
| `pytest.ini` | Points `pytest` at `tests/` |
| `CLAUDE.md` | Conventions and rules for AI coding assistants working in this repo |

## Design decisions

- **Custom tool node, not the prebuilt `ToolNode`.** Tools run one after another, so the 0.3 s politeness delay
  really holds and SQLite is only touched from one thread. The agent can still ask for many product pages in one turn,
  which keeps the number of LLM round-trips small (roughly 15 turns per run).
- **Tools never raise.** A `@logged` decorator catches every exception, turns it into `{"ok": false, "error": ...}`,
  and writes the call, a shortened result and the agent's reason to the console and `agent_logs`.
- **"Reason" = the agent's own words.** The system prompt asks the model to write one sentence before each tool call.
  That text is logged as the reason, so the tool signatures stay exactly as specified.
- **Loop safety.** `recursion_limit=200` graph steps, at most 3 retries per URL (enforced inside `retry_failed`,
  not just in the prompt), and the graph ends as soon as `finish` is called or the model stops calling tools.
- **Retry strategies.** `backoff` waits 1 s, 2 s, 4 s before retrying. `alternate_user_agent` retries straight away
  with a different User-Agent. The prompt suggests backoff first for 5xx/timeouts, then switching, but the model picks.
- **Bronze keeps raw strings** (`£51.77`, `star-rating Three`, `In stock (22 available)`). All parsing and casting
  happens on the way to Silver, so a parser change can be replayed from Bronze.
- **Silver is idempotent.** It has `PRIMARY KEY(product_url)` and is written with `INSERT ... ON CONFLICT DO UPDATE`,
  so reruns update rows instead of adding new ones. Gold is fully rebuilt from Silver on each save.
- **Gemini instead of Claude.** The spec asked for `langchain-anthropic`, but the project uses Google Gemini
  (`langchain-google-genai`) because it has a free tier. Only `get_llm()` in `src/agent.py` knows about the provider,
  so switching back means changing that one function. Calls are paced to stay under the free tier's
  5 requests/minute, and a fallback chain of Gemini models (each with its own daily quota) takes over when
  a model's quota runs out or it times out.
- **Grounded summary.** `get_facts` reads only from Gold, and the prompt says the summary may use only those numbers.

## Assumptions and simplifications

- **"Append-only" Bronze:** every run appends a fresh set of rows tagged with its `run_id`. `scrape_product_page`
  fills `category_raw` / `stock_raw` on *that run's* rows (the spec says "update Bronze"). Rows from earlier runs
  are never changed or deleted.
- **Invalid rows are dropped, not repaired.** A row missing price, category or stock, with a rating outside 1–5,
  or with a non-books.toscrape.com URL is left out of Silver, and `save_clean_data` reports why. If a URL still fails
  after 3 retries, the run finishes with fewer than 60 books.
- **Duplicates:** the same URL scraped twice in one run is collapsed, and the latest scrape wins.
- **Gold is computed over all of Silver** (currently the same 60 books), not only the current run.
- **Simulated failure** is an HTTP 503 (the spec allowed a timeout or a 503). It fires on the first request to page 2 only.
- **`gold_summary`** keeps only the latest summary.
- **Run state** (pages done, failed URLs, retry counts) lives in memory for the duration of one run.
  Only one run at a time is expected.
- Only the 3 listing pages and their 60 product pages are fetched, with a 10 s timeout and a descriptive User-Agent.
