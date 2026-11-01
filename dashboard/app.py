"""Streamlit dashboard: reads Silver/Gold tables from data/books.db and can trigger a pipeline run.

Run with:  streamlit run dashboard/app.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

st.set_page_config(page_title="Books Agent Dashboard", page_icon="📚", layout="wide")


def read_sql(sql: str) -> pd.DataFrame:
    """Read a query into a DataFrame; empty frame if the DB or table doesn't exist yet."""
    if not config.DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()


st.title("📚 Agentic Books Pipeline")
st.caption(f"Source: books.toscrape.com (pages {config.PAGES_TO_SCRAPE}) · Model: {config.MODEL} · DB: {config.DB_PATH.name}")

# --- Run controls -------------------------------------------------------------
col_btn, col_chk, _ = st.columns([1, 1, 3])
simulate = col_chk.checkbox("Simulate failure", value=config.simulate_failure_from_env(),
                            help="First request to page 2 fails once with a simulated HTTP 503")
if col_btn.button("▶ Run Pipeline", type="primary"):
    import os
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini key.")
    else:
        from src.agent import run_agent
        with st.spinner("Agent is running (scraping 60 books politely takes ~1-2 minutes)..."):
            result = run_agent(simulate_failure=simulate)
        (st.success if result["finished"] else st.warning)(f"{result['run_id']}: {result['reason']}")

kpis = read_sql("SELECT * FROM gold_kpis")
if kpis.empty:
    st.info("No data yet. Click **Run Pipeline** (or run `python -m src.run_pipeline`).")
    st.stop()

# --- KPIs ---------------------------------------------------------------------
k = kpis.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total books scraped", int(k["total_books"]))
c2.metric("Average price", f"£{k['avg_price']:.2f}")
c3.metric("Average rating", f"{k['avg_rating']:.2f} / 5")
c4.metric("Total stock", int(k["total_stock"]))

# --- AI summary -----------------------------------------------------------------
st.subheader("🤖 AI-generated summary")
summary = read_sql("SELECT summary, created_at FROM gold_summary ORDER BY created_at DESC LIMIT 1")
if summary.empty:
    st.write("_No summary stored yet._")
else:
    st.info(summary.iloc[0]["summary"])
    st.caption(f"Generated {summary.iloc[0]['created_at']} from numbers returned by get_facts")

# --- Charts -------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Books by category")
    cats = read_sql("SELECT category, book_count FROM gold_books_by_category ORDER BY book_count DESC")
    st.bar_chart(cats, x="category", y="book_count", horizontal=True, sort="-book_count")
with right:
    st.subheader("Rating distribution")
    ratings = read_sql("SELECT rating || ' ★' AS rating, book_count FROM gold_rating_distribution ORDER BY rating")
    st.bar_chart(ratings, x="rating", y="book_count")

# --- Top 10 -------------------------------------------------------------------
st.subheader("Top 10 most expensive books")
top10 = read_sql("SELECT rank, title, price_gbp, rating, category, product_url FROM gold_top10_expensive ORDER BY rank")
st.dataframe(
    top10, hide_index=True, width="stretch",
    column_config={
        "price_gbp": st.column_config.NumberColumn("Price (£)", format="£%.2f"),
        "product_url": st.column_config.LinkColumn("Link", display_text="open"),
    },
)

# --- Agent logs -----------------------------------------------------------------
st.subheader("Agent execution log (latest run)")
logs = read_sql(
    """SELECT timestamp AS time, tool, status, details FROM agent_logs
       WHERE run_id = (SELECT run_id FROM agent_logs ORDER BY id DESC LIMIT 1)
       ORDER BY id"""
)

# Colour failures red, retries amber, recoveries green
HIGHLIGHT = {"ERROR": "#f8d7da", "FAILURE_INJECTED": "#f8d7da", "GAVE_UP": "#f8d7da",
             "RETRY": "#fff3cd", "RECOVERED": "#d1e7dd"}


def highlight(row):
    colour = HIGHLIGHT.get(row["status"])
    return [f"background-color: {colour}; color: #000" if colour else ""] * len(row)


if logs.empty:
    st.write("_No logs yet._")
else:
    events = logs["status"].value_counts()
    st.caption(" · ".join(f"{s}: {n}" for s, n in events.items()))
    only_issues = st.toggle("Show only failures / retries / recoveries", value=False)
    view = logs[logs["status"].isin(HIGHLIGHT)] if only_issues else logs
    st.dataframe(view.style.apply(highlight, axis=1), hide_index=True, width="stretch", height=420)
