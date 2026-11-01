"""Streamlit dashboard: reads Silver/Gold tables from data/books.db and can trigger a pipeline run.

Run with:  streamlit run dashboard/app.py
"""
import html
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# On Streamlit Community Cloud there is no .env: copy secrets into env vars before loading config
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:  # no secrets.toml locally is fine (.env is used instead)
    pass

# Make `src` importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import agent, config, tools  # noqa: E402

st.set_page_config(page_title="Books Pipeline", layout="wide", initial_sidebar_state="auto")

# --------------------------------------------------------------------------- #
# Design tokens: one blue for data, status colours only for pipeline state
# --------------------------------------------------------------------------- #
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
CARD, BORDER, GRID = "#ffffff", "#e7e6e2", "#eeede9"
SERIES = "#2a78d6"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
# Log statuses to highlight: status -> (dot colour, row tint)
HIGHLIGHT = {
    "ERROR": (CRITICAL, "#fbeaea"), "FAILURE_INJECTED": (CRITICAL, "#fbeaea"), "GAVE_UP": (CRITICAL, "#fbeaea"),
    "RETRY": ("#fab219", "#fef5e1"), "RECOVERED": (GOOD, "#e6f5e6"),
}

st.markdown(f"""
<style>
  #MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

  .page-head {{ display: flex; justify-content: space-between; align-items: flex-end;
               flex-wrap: wrap; gap: 12px; margin-bottom: 1.4rem; }}
  .eyebrow {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
             color: {SERIES}; margin-bottom: 4px; }}
  .page-title {{ font-size: 2rem; font-weight: 700; color: {INK}; line-height: 1.15; margin: 0; padding: 0; }}
  .page-sub {{ color: {INK_2}; margin-top: 6px; font-size: 0.95rem; }}

  .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px;
          border: 1px solid {BORDER}; background: {CARD}; color: {INK}; font-size: 0.83rem; white-space: nowrap; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }}

  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 1.4rem; }}
  @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 20px; }}
  .kpi-label {{ font-size: 0.8rem; color: {INK_2}; font-weight: 500; }}
  .kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {INK}; margin: 6px 0 2px;
               font-variant-numeric: tabular-nums; }}
  .kpi-note {{ font-size: 0.8rem; color: {INK_3}; }}

  .card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px 22px; }}
  .summary {{ border-left: 3px solid {SERIES}; }}
  .summary-text {{ color: {INK}; font-size: 1.02rem; line-height: 1.65; margin: 8px 0 10px; }}
  .section-label {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                   color: {INK_2}; }}
  .muted {{ color: {INK_3}; font-size: 0.8rem; }}

  .chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 14px; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
  .side-meta {{ font-size: 0.82rem; color: {INK_2}; line-height: 1.9; margin-top: 18px; }}
  .side-meta b {{ color: {INK}; font-weight: 600; }}

  .live {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 18px 20px; margin-bottom: 1.4rem; }}
  .live-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }}
  .live-title {{ font-weight: 650; color: {INK}; font-size: 1.02rem; }}
  .steps {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }}
  @media (max-width: 900px) {{ .steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .step {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 12px; }}
  .step-name {{ font-size: 0.8rem; color: {INK_2}; display: flex; align-items: center; gap: 8px; }}
  .step-val {{ font-size: 0.95rem; font-weight: 600; color: {INK}; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .step.active {{ border-color: {SERIES}; box-shadow: 0 0 0 1px {SERIES} inset; }}
  .bar {{ height: 6px; background: {GRID}; border-radius: 999px; overflow: hidden; margin: 4px 0 16px; }}
  .bar > div {{ height: 100%; background: {SERIES}; border-radius: 999px; transition: width .4s; }}
  .feed-row {{ display: grid; grid-template-columns: 64px 104px 1fr; gap: 10px; align-items: start;
              padding: 7px 0; border-bottom: 1px solid {GRID}; font-size: 0.86rem; color: {INK}; }}
  .feed-row:last-child {{ border-bottom: none; }}
  .feed-time {{ color: {INK_3}; font-variant-numeric: tabular-nums; }}
  .feed-text {{ overflow-wrap: anywhere; }}
  .feed-row .pill {{ padding: 2px 9px; font-size: 0.76rem; }}
  .term {{ background: #111214; color: #d7d6d2; border-radius: 10px; padding: 14px 16px; margin: 0;
          font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; height: 420px; overflow: hidden;
          white-space: pre-wrap; overflow-wrap: anywhere; display: flex; flex-direction: column; justify-content: flex-end; }}
  .term .t-err {{ color: #ff8a8a; }} .term .t-retry {{ color: #ffd27a; }} .term .t-ok {{ color: #8fdc8f; }}
  .term .t-agent {{ color: #8ab8f0; }}
</style>
""", unsafe_allow_html=True)


def read_sql(sql: str) -> pd.DataFrame:
    """Read a query into a DataFrame; empty frame if the DB or table doesn't exist yet."""
    if not config.DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()


def pill(text: str, color: str) -> str:
    """Small rounded label: coloured dot carries the meaning, text stays in ink."""
    return f'<span class="pill"><span class="dot" style="background:{color}"></span>{html.escape(text)}</span>'


def bar_chart(df: pd.DataFrame, cat: str, value: str, horizontal: bool, cat_title: str) -> alt.Chart:
    """Single-series bar chart: one hue, rounded data ends, recessive axes, hover tooltip."""
    value_axis = alt.Axis(grid=True, gridColor=GRID, domain=False, ticks=False, tickMinStep=1,
                          labelColor=INK_2, titleColor=INK_2, titlePadding=10)
    cat_axis = alt.Axis(grid=False, domain=False, ticks=False, labelColor=INK, titleColor=INK_2,
                        labelLimit=180, labelPadding=8, labelAngle=0)
    band = alt.Scale(paddingInner=0.3)
    if horizontal:
        enc = dict(y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=cat_axis, scale=band),
                   x=alt.X(f"{value}:Q", title="Books", axis=value_axis))
    else:
        enc = dict(x=alt.X(f"{cat}:O", title=cat_title, axis=cat_axis, scale=band),
                   y=alt.Y(f"{value}:Q", title="Books", axis=value_axis))
    return (
        alt.Chart(df)
        .mark_bar(color=SERIES, cornerRadiusEnd=4)
        .encode(**enc, tooltip=[alt.Tooltip(f"{cat}:N", title=cat_title), alt.Tooltip(f"{value}:Q", title="Books")])
        .properties(height=max(280, 26 * len(df)) if horizontal else 320)
        .configure_view(stroke=None)
        .configure(background="transparent")
    )


# --------------------------------------------------------------------------- #
# Sidebar: run controls and pipeline details
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Pipeline")
    st.caption("A LangGraph agent decides every step: scrape, retry, validate, save and summarise.")
    simulate = st.toggle("Simulate failure", value=config.simulate_failure_from_env(),
                         help="The first request to page 2 fails once with a simulated HTTP 503. "
                              "Watch the agent retry and recover in the Agent log tab.")
    run_clicked = st.button("Run pipeline", type="primary", width="stretch", disabled=agent._RUN_LOCK.locked(),
                            help="Disabled while a run is in progress")
    st.markdown(
        f"""<div class="side-meta">
        <b>Source</b> books.toscrape.com<br>
        <b>Pages</b> {', '.join(map(str, config.PAGES_TO_SCRAPE))} ({20 * len(config.PAGES_TO_SCRAPE)} books)<br>
        <b>Model</b> {html.escape(config.MODEL)}<br>
        <b>Fallbacks</b> {len(config.FALLBACK_MODELS)} models<br>
        <b>Storage</b> SQLite, Bronze / Silver / Gold</div>""",
        unsafe_allow_html=True,
    )
    st.caption("A run takes about 2 to 5 minutes on the Gemini free tier.")

# --------------------------------------------------------------------------- #
# Header with last-run status
# --------------------------------------------------------------------------- #
last = read_sql("""SELECT run_id, timestamp,
                          (SELECT COUNT(*) FROM agent_logs e WHERE e.run_id = l.run_id
                             AND e.tool = 'pipeline' AND e.status IN ('ERROR', 'STOPPED')) AS errors
                   FROM agent_logs l WHERE tool = 'pipeline' AND status = 'END' ORDER BY id DESC LIMIT 1""")
if last.empty:
    head_pill = pill("No runs yet", INK_3)
else:
    ok = int(last.iloc[0]["errors"]) == 0
    when = last.iloc[0]["timestamp"].replace("T", " ")[:16] + " UTC"
    head_pill = pill(f"Last run {'succeeded' if ok else 'stopped'} · {when}", GOOD if ok else CRITICAL)

st.markdown(f"""
<div class="page-head">
  <div>
    <div class="eyebrow">Agentic data pipeline</div>
    <div class="page-title">Books Catalogue Insights</div>
    <div class="page-sub">Prices, ratings, categories and stock for the first
      {20 * len(config.PAGES_TO_SCRAPE)} books on books.toscrape.com</div>
  </div>
  <div>{head_pill}</div>
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Live run panel: the agent runs in a background thread, the page redraws every second
# --------------------------------------------------------------------------- #
TAGS = {  # event status -> (label, dot colour)
    "FAILURE_INJECTED": ("Failure", CRITICAL), "ERROR": ("Error", CRITICAL), "GAVE_UP": ("Gave up", CRITICAL),
    "RETRY": ("Retry", "#fab219"), "RECOVERED": ("Recovered", GOOD), "STOPPED": ("Stopped", "#fab219"),
    "START": ("Run", INK_3), "END": ("Run", INK_3),
}
PINNED = {"FAILURE_INJECTED", "ERROR", "GAVE_UP", "RETRY", "RECOVERED"}  # stay in the feed for the whole run


def tag_for(event: dict) -> tuple[str, str]:
    if event["source"] == "agent":
        return "Agent", SERIES
    return TAGS.get(event["status"], ("Tool", INK_3))


def term_class(line: str) -> str:
    if "AGENT:" in line or "[model:" in line or "-> decides" in line:
        return "t-agent"
    for word, cls in [("FAILURE_INJECTED", "t-err"), (" ERROR ", "t-err"), ("GAVE_UP", "t-err"),
                      ("RETRY", "t-retry"), ("RECOVERED", "t-ok")]:
        if word in line:
            return cls
    return ""


def draw_live(panel, events: list[dict], running: bool, result: dict | None = None) -> None:
    """Step tracker, progress bar, readable activity feed and the raw console output."""
    done_tools = {e["source"] for e in events if e["status"] == "OK"}
    pages = len(tools.STATE.pages_done)
    products = sum(e["source"] == "scrape_product_page" and e["status"] == "OK" for e in events)
    total_books = 20 * len(config.PAGES_TO_SCRAPE)
    steps = [
        ("Listing pages", f"{pages} / {len(config.PAGES_TO_SCRAPE)}", pages == len(config.PAGES_TO_SCRAPE)),
        ("Product pages", f"{products} / {total_books}", products >= total_books or "validate_data" in done_tools),
        ("Validate", "done" if "validate_data" in done_tools else "waiting", "validate_data" in done_tools),
        ("Save", "done" if "save_clean_data" in done_tools else "waiting", "save_clean_data" in done_tools),
        ("Summary", "done" if "save_summary" in done_tools else "waiting", "save_summary" in done_tools),
    ]
    active = next((i for i, s in enumerate(steps) if not s[2]), None) if running else None
    progress = (pages + products + 10 * sum(s[2] for s in steps[2:])) / (len(config.PAGES_TO_SCRAPE) + total_books + 30)

    finished = result["finished"] if result else "finish" in done_tools  # result is None when watching another run
    if running and tools.STOP_REQUESTED.is_set():
        head = pill("Stopping after the current step", "#fab219")
    elif running:
        head = pill("Running", SERIES)
    elif finished:
        head = pill("Complete", GOOD)
    else:
        head = pill("Stopped", CRITICAL)
    step_html = "".join(
        f'<div class="step{" active" if i == active else ""}"><div class="step-name">'
        f'<span class="dot" style="background:{GOOD if ok else (SERIES if i == active else BORDER)}"></span>{name}</div>'
        f'<div class="step-val">{val}</div></div>' for i, (name, val, ok) in enumerate(steps))
    # Newest 12 events, plus any earlier failures/retries/recoveries so they never scroll away
    visible = [e for e in events if (e["source"] != "fetch" or e["status"] != "OK") and e["status"] != "END"]
    shown = [e for e in visible[:-12] if e["status"] in PINNED] + visible[-12:]
    feed_html = "".join(
        f'<div class="feed-row"><span class="feed-time">{e["time"]}</span>{pill(*tag_for(e))}'
        f'<span class="feed-text">{html.escape(e["text"][:220])}</span></div>'
        for e in reversed(shown))
    lines = "\n".join(e["console"] for e in events).strip().splitlines()[-60:]
    term_html = "".join(f'<div class="{term_class(l)}">{html.escape(l)}</div>' for l in lines)

    with panel.container():
        st.markdown(
            f'<div class="live"><div class="live-head"><div class="live-title">Agent activity</div>{head}</div>'
            f'<div class="steps">{step_html}</div>'
            f'<div class="bar"><div style="width:{min(100, round(100 * progress))}%"></div></div></div>',
            unsafe_allow_html=True)
        feed_col, term_col = st.columns([1, 1], gap="medium")
        with feed_col:
            st.markdown('<div class="section-label">What the agent is doing (newest first)</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:6px">{feed_html or "<span class=muted>Starting...</span>"}</div>',
                        unsafe_allow_html=True)
        with term_col:
            st.markdown('<div class="section-label">Console output</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="term" style="margin-top:6px">{term_html}</div>', unsafe_allow_html=True)
        if not running:
            # Reason comes from our own run's result, or from the END log line when watching another run
            ends = [e["text"] for e in events if e["source"] == "pipeline" and e["status"] == "END"]
            reason = result["reason"] if result else (ends[-1] if ends else "")
            (st.success if finished else st.warning)(f"{'Run complete' if finished else 'Run stopped'}: {reason}")


thread = None
result_box: dict = {}
if run_clicked:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Add it to .env locally, or to the app's Secrets when hosted.")
    elif agent._RUN_LOCK.locked():
        st.warning("A run is already in progress. Its live activity is shown below.")
    else:
        tools.EVENTS.clear()
        st.session_state.pop("last_run", None)
        thread = threading.Thread(target=lambda: result_box.update(result=agent.run_agent(simulate_failure=simulate)),
                                  daemon=True)
        thread.start()

# Show the live panel for our own run, or for a run someone else started (lock held)
if thread is not None or agent._RUN_LOCK.locked():
    stop_col, note_col = st.columns([1, 5], vertical_alignment="center")
    if stop_col.button("Stop run", key="stop_run", width="stretch"):
        tools.STOP_REQUESTED.set()
    note_col.caption("Stopping ends the run after the current step (a model call can take a few seconds). "
                     "Data already saved stays valid; reruns never create duplicates.")
    panel = st.empty()
    while (thread is not None and thread.is_alive()) or (thread is None and agent._RUN_LOCK.locked()):
        draw_live(panel, list(tools.EVENTS), running=True)
        time.sleep(1)
    # Keep the final panel for this visitor, then redraw the page so the Run button and data refresh
    st.session_state["last_run"] = {"events": list(tools.EVENTS), "result": result_box.get("result")}
    st.rerun()
elif "last_run" in st.session_state:
    draw_live(st.empty(), st.session_state["last_run"]["events"], running=False,
              result=st.session_state["last_run"]["result"])

kpis = read_sql("SELECT * FROM gold_kpis")
if kpis.empty:
    st.markdown('<div class="card"><div class="section-label">No data yet</div>'
                '<p class="summary-text">Click <b>Run pipeline</b> in the sidebar, or run '
                '<code>python -m src.run_pipeline</code>.</p></div>', unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
k = kpis.iloc[0]
cards = [
    ("Total books scraped", f"{int(k['total_books'])}", f"of {20 * len(config.PAGES_TO_SCRAPE)} expected"),
    ("Average price", f"£{k['avg_price']:.2f}", f"range £{k['min_price']:.2f} to £{k['max_price']:.2f}"),
    ("Average rating", f"{k['avg_rating']:.2f}", "out of 5"),
    ("Units in stock", f"{int(k['total_stock']):,}", "across all books"),
]
st.markdown('<div class="kpi-grid">' + "".join(
    f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
    f'<div class="kpi-note">{note}</div></div>' for label, value, note in cards) + "</div>",
    unsafe_allow_html=True)

tab_overview, tab_books, tab_log = st.tabs(["Overview", "All books", "Agent log"])

# --------------------------------------------------------------------------- #
# Overview: summary, charts, top 10
# --------------------------------------------------------------------------- #
with tab_overview:
    summary = read_sql("SELECT summary, created_at FROM gold_summary ORDER BY created_at DESC LIMIT 1")
    if not summary.empty:
        created = summary.iloc[0]["created_at"].replace("T", " ")[:16]
        st.markdown(
            f'<div class="card summary"><div class="section-label">AI-generated summary</div>'
            f'<p class="summary-text">{html.escape(summary.iloc[0]["summary"])}</p>'
            f'<div class="muted">Written by the agent at {created} UTC, using only numbers returned by get_facts</div>'
            f'</div>', unsafe_allow_html=True)
        st.write("")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Books by category")
        cats = read_sql("SELECT category, book_count FROM gold_books_by_category ORDER BY book_count DESC, category")
        st.altair_chart(bar_chart(cats.head(10), "category", "book_count", True, "Category"), width="stretch")
        if len(cats) > 10:
            st.caption(f"Top 10 of {len(cats)} categories. The All books tab lists every category.")
    with right:
        st.markdown("#### Rating distribution")
        ratings = read_sql("SELECT rating || ' / 5' AS rating, book_count FROM gold_rating_distribution ORDER BY rating")
        st.altair_chart(bar_chart(ratings, "rating", "book_count", False, "Rating"), width="stretch")

    st.markdown("#### Top 10 most expensive books")
    top10 = read_sql("SELECT rank, title, category, rating, price_gbp, product_url "
                     "FROM gold_top10_expensive ORDER BY rank")
    st.dataframe(
        top10, hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "category": "Category",
            "rating": st.column_config.NumberColumn("Rating", format="%d / 5"),
            "price_gbp": st.column_config.ProgressColumn(
                "Price", format="£%.2f", min_value=0, max_value=float(top10["price_gbp"].max())),
            "product_url": st.column_config.LinkColumn("Link", display_text="View"),
        },
    )

# --------------------------------------------------------------------------- #
# All books: searchable Silver table
# --------------------------------------------------------------------------- #
with tab_books:
    books = read_sql("SELECT title, category, price_gbp, rating, stock_qty, product_url FROM silver_books "
                     "ORDER BY title")
    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Search titles", placeholder="Type to filter")
    chosen = c2.multiselect("Category", sorted(books["category"].unique()))
    view = books
    if query:
        view = view[view["title"].str.contains(query, case=False, regex=False)]
    if chosen:
        view = view[view["category"].isin(chosen)]
    st.caption(f"{len(view)} of {len(books)} books · Silver table, one row per product URL")
    st.dataframe(
        view, hide_index=True, width="stretch", height=520,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "category": "Category",
            "price_gbp": st.column_config.NumberColumn("Price", format="£%.2f"),
            "rating": st.column_config.NumberColumn("Rating", format="%d / 5"),
            "stock_qty": st.column_config.NumberColumn("In stock"),
            "product_url": st.column_config.LinkColumn("Link", display_text="View"),
        },
    )

# --------------------------------------------------------------------------- #
# Agent log for the latest run, failures and retries highlighted
# --------------------------------------------------------------------------- #
with tab_log:
    logs = read_sql(
        """SELECT timestamp AS time, tool, status, details FROM agent_logs
           WHERE run_id = (SELECT run_id FROM agent_logs ORDER BY id DESC LIMIT 1)
           ORDER BY id"""
    )
    if logs.empty:
        st.caption("No logs yet.")
    else:
        logs["time"] = logs["time"].str.slice(11, 19)
        counts = logs["status"].value_counts()
        tool_calls = int((logs["status"].isin(["OK", "ERROR"]) & ~logs["tool"].isin(["pipeline", "fetch"])).sum())
        chips = [pill(f"{tool_calls} tool calls", SERIES)]
        for name, label in [("FAILURE_INJECTED", "failure injected"), ("ERROR", "errors"),
                            ("RETRY", "retries"), ("RECOVERED", "recovered"), ("GAVE_UP", "gave up")]:
            if counts.get(name):
                chips.append(pill(f"{counts[name]} {label}", HIGHLIGHT[name][0]))
        st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)

        only_issues = st.toggle("Show only failures, retries and recoveries")
        view = logs[logs["status"].isin(HIGHLIGHT)] if only_issues else logs

        def highlight(row):
            tint = HIGHLIGHT.get(row["status"], (None, None))[1]
            return [f"background-color: {tint}; color: {INK}" if tint else ""] * len(row)

        st.dataframe(
            view.style.apply(highlight, axis=1), hide_index=True, width="stretch", height=480,
            column_config={
                "time": st.column_config.TextColumn("Time (UTC)", width="small"),
                "tool": st.column_config.TextColumn("Tool", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "details": st.column_config.TextColumn("Details", width="large"),
            },
        )
