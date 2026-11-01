"""SQLite schema (Bronze -> Silver -> Gold + agent_logs) and small helpers."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src import config

SCHEMA = """
-- Bronze: raw scraped strings, append-only across runs
CREATE TABLE IF NOT EXISTS bronze_books (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    scraped_at   TEXT NOT NULL,
    source_page  TEXT NOT NULL,
    title        TEXT,
    price_raw    TEXT,
    rating_raw   TEXT,
    product_url  TEXT,
    category_raw TEXT,
    stock_raw    TEXT
);

-- Silver: cleaned + typed, one row per book
CREATE TABLE IF NOT EXISTS silver_books (
    product_url TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL NOT NULL,
    rating      INTEGER NOT NULL,
    category    TEXT NOT NULL,
    stock_qty   INTEGER NOT NULL,
    run_id      TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Gold: aggregates for the dashboard, rebuilt on each run
CREATE TABLE IF NOT EXISTS gold_kpis (
    run_id       TEXT,
    total_books  INTEGER,
    avg_price    REAL,
    min_price    REAL,
    max_price    REAL,
    avg_rating   REAL,
    total_stock  INTEGER,
    built_at     TEXT
);
CREATE TABLE IF NOT EXISTS gold_books_by_category (
    category   TEXT,
    book_count INTEGER,
    avg_price  REAL
);
CREATE TABLE IF NOT EXISTS gold_rating_distribution (
    rating     INTEGER,
    book_count INTEGER
);
CREATE TABLE IF NOT EXISTS gold_top10_expensive (
    rank        INTEGER,
    title       TEXT,
    price_gbp   REAL,
    rating      INTEGER,
    category    TEXT,
    product_url TEXT
);
CREATE TABLE IF NOT EXISTS gold_summary (
    run_id     TEXT,
    created_at TEXT,
    summary    TEXT
);

-- Every tool call / event the agent produces
CREATE TABLE IF NOT EXISTS agent_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    timestamp TEXT,
    tool      TEXT,
    status    TEXT,
    details   TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    """Open a connection to the configured DB, commit on success, always close."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def insert_log(run_id: str, tool: str, status: str, details: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO agent_logs (run_id, timestamp, tool, status, details) VALUES (?, ?, ?, ?, ?)",
            (run_id, now_iso(), tool, status, details),
        )


def insert_bronze_rows(run_id: str, source_page: str, rows: list[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            """INSERT INTO bronze_books
               (run_id, scraped_at, source_page, title, price_raw, rating_raw, product_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(run_id, now_iso(), source_page, r["title"], r["price_raw"], r["rating_raw"], r["product_url"])
             for r in rows],
        )


def update_bronze_details(run_id: str, product_url: str, category_raw: str, stock_raw: str) -> int:
    """Fill in product-page fields on this run's Bronze rows. Returns rows updated."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE bronze_books SET category_raw = ?, stock_raw = ? WHERE run_id = ? AND product_url = ?",
            (category_raw, stock_raw, run_id, product_url),
        )
        return cur.rowcount


def bronze_rows(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM bronze_books WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def query(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
