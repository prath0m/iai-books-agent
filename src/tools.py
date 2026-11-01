"""Agent tools. Every tool is a LangChain @tool that never raises and returns {"ok": bool, ...}.

Each tool call is logged (console + agent_logs table) by the @logged decorator.
"""
import functools
import json
import re
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src import config, db

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


# --------------------------------------------------------------------------- #
# Run state (one pipeline run at a time)
# --------------------------------------------------------------------------- #
@dataclass
class RunState:
    run_id: str = "manual"
    simulate_failure: bool = False
    failure_injected: bool = False
    pages_done: set = field(default_factory=set)
    failed: dict = field(default_factory=dict)        # url -> last error
    retry_counts: dict = field(default_factory=dict)  # url -> retries used
    current_reason: str = ""                          # set by the agent before each tool call
    finished: bool = False
    finish_reason: str = ""


STATE = RunState()

# Live feed of the current run (read by the dashboard while the agent works).
# Each item: {"time", "source", "status", "text", "console"}; "text" is a short plain-English line.
EVENTS: list[dict] = []

# Set by the dashboard's Stop button; the agent checks it before every model call and tool call
STOP_REQUESTED = threading.Event()


def start_run(run_id: str, simulate_failure: bool = False) -> None:
    """Reset state for a new run and make sure the DB schema exists."""
    global STATE
    STATE = RunState(run_id=run_id, simulate_failure=simulate_failure)
    EVENTS.clear()
    STOP_REQUESTED.clear()
    db.init_db()


def emit(source: str, status: str, text: str, console: str) -> None:
    """Print a console line and add it to the live feed."""
    print(console, flush=True)
    EVENTS.append({"time": db.now_iso()[11:19], "source": source, "status": status, "text": text, "console": console})


def log_event(tool_name: str, status: str, details: str, text: str | None = None) -> None:
    """Write one log line to the console, the live feed and agent_logs."""
    emit(tool_name, status, text or details, f"[{db.now_iso()}] {tool_name:<20} {status:<16} {details}")
    try:
        db.insert_log(STATE.run_id, tool_name, status, details)
    except Exception as exc:  # logging must never break the pipeline
        print(f"  (could not write log to DB: {exc})")


def describe(tool_name: str, args: dict, result: dict) -> str:
    """One readable sentence about a tool call, for the live feed."""
    if not result.get("ok"):
        return f"{tool_name} failed: {result.get('error', 'unknown error')}"
    if tool_name == "scrape_listing_page":
        return f"Listing page {result['page']}: found {result['books_found']} books"
    if tool_name == "scrape_product_page":
        slug = result["product_url"].rstrip("/").split("/")[-2].rsplit("_", 1)[0].replace("-", " ")
        return f"Product '{slug[:45]}': {result['category']}, {result['stock']}"
    if tool_name == "scrape_product_pages":
        return f"Batch done: {result['fetched']} product pages fetched"
    if tool_name == "get_status":
        return (f"Status check: pages done {result['pages_done']}, "
                f"{result['books_missing_category_or_stock']} books missing details, "
                f"{len(result['failed_urls'])} failed URLs")
    if tool_name == "retry_failed":
        return f"Retry succeeded on attempt {result['attempt']} using {result['strategy']}"
    if tool_name == "validate_data":
        return f"Validation: {result['total_issues']} issues in {result['report']['rows_checked']} rows"
    if tool_name == "save_clean_data":
        return f"Saved {result['rows_upserted']} rows to Silver ({result['silver_row_count']} in total), Gold rebuilt"
    if tool_name == "get_facts":
        return f"Facts: {result['total_books']} books, average price £{result['average_price_gbp']}"
    if tool_name == "save_summary":
        return "AI summary saved"
    if tool_name == "finish":
        return f"Finished: {result['reason']}"
    return f"{tool_name} done"


def logged(func):
    """Wrap a tool body: catch all exceptions and log the call, its result and the agent's reason."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        status = "OK" if result.get("ok") else "ERROR"
        short = json.dumps(result, default=str)
        short = short if len(short) <= 300 else short[:300] + "..."
        reason = f" | reason: {STATE.current_reason}" if STATE.current_reason else ""
        log_event(func.__name__, status, f"args={json.dumps(kwargs, default=str)} result={short}{reason}",
                  text=describe(func.__name__, kwargs, result))
        return result
    return wrapper


# --------------------------------------------------------------------------- #
# Parsing helpers (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def parse_price(text: str | None) -> float | None:
    """'£51.77' (or mis-decoded 'Â£51.77') -> 51.77. None if no number."""
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def parse_rating(css_class: str | list | None) -> int | None:
    """'star-rating Three' or ['star-rating', 'Three'] -> 3. None if unknown."""
    if not css_class:
        return None
    words = css_class.split() if isinstance(css_class, str) else css_class
    for word in words:
        if word in RATING_WORDS:
            return RATING_WORDS[word]
    return None


def parse_stock(text: str | None) -> int | None:
    """'In stock (22 available)' -> 22. None if not found."""
    if not text:
        return None
    match = re.search(r"\((\d+)\s+available\)", text)
    return int(match.group(1)) if match else None


def parse_listing(html: str | bytes, page_url: str) -> list[dict]:
    """Extract raw title/price/rating/url for every book on a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for pod in soup.select("article.product_pod"):
        link = pod.select_one("h3 a")
        rating_tag = pod.select_one("p.star-rating")
        price_tag = pod.select_one("p.price_color")
        rows.append({
            "title": link.get("title") or link.get_text(strip=True),
            "price_raw": price_tag.get_text(strip=True) if price_tag else None,
            # keep the raw class text, e.g. "star-rating Three"; parsed in Silver
            "rating_raw": " ".join(rating_tag.get("class", [])) if rating_tag else None,
            "product_url": urljoin(page_url, link["href"]),
        })
    return rows


def parse_product(html: str | bytes) -> dict:
    """Extract raw category (breadcrumb) and stock text from a product page."""
    soup = BeautifulSoup(html, "html.parser")
    crumbs = [li.get_text(strip=True) for li in soup.select("ul.breadcrumb li")]
    # Breadcrumb is: Home > Books > <Category> > <Title>
    category = crumbs[2] if len(crumbs) >= 4 else None
    stock_tag = soup.select_one("p.instock.availability")
    stock = " ".join(stock_tag.get_text().split()) if stock_tag else None
    return {"category_raw": category, "stock_raw": stock}


def is_valid_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc == "books.toscrape.com"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def fetch(url: str, user_agent: str = config.USER_AGENT) -> dict:
    """Polite GET. Returns {"ok": True, "html": bytes} or {"ok": False, "error": str}. Never raises."""
    # Simulated failure: first request to page 2 fails once with an HTTP 503
    if STATE.simulate_failure and not STATE.failure_injected and url.endswith("page-2.html"):
        STATE.failure_injected = True
        log_event("fetch", "FAILURE_INJECTED", f"Simulated HTTP 503 for {url}")
        return {"ok": False, "error": "HTTP 503 Service Unavailable (simulated)"}

    time.sleep(config.REQUEST_DELAY_SECONDS)
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=config.REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}"}
    return {"ok": True, "html": resp.content}  # bytes: BeautifulSoup reads the page's own charset


def _page_url(page_number: int) -> str:
    return config.LISTING_URL.format(page=page_number)


def _ingest_listing(page_number: int, html: bytes) -> dict:
    url = _page_url(page_number)
    rows = parse_listing(html, url)
    db.insert_bronze_rows(STATE.run_id, url, rows)
    STATE.pages_done.add(page_number)
    STATE.failed.pop(url, None)
    return {"ok": True, "page": page_number, "books_found": len(rows),
            "product_urls": [r["product_url"] for r in rows]}


def _ingest_product(product_url: str, html: bytes) -> dict:
    details = parse_product(html)
    updated = db.update_bronze_details(STATE.run_id, product_url, details["category_raw"], details["stock_raw"])
    STATE.failed.pop(product_url, None)
    return {"ok": True, "product_url": product_url, "category": details["category_raw"],
            "stock": details["stock_raw"], "bronze_rows_updated": updated}


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool
@logged
def scrape_listing_page(page_number: int) -> dict:
    """Fetch and parse one catalogue listing page (1-3). Saves raw rows (title, price, rating, url) to Bronze
    and returns the product URLs found on the page."""
    if page_number not in config.PAGES_TO_SCRAPE:
        return {"ok": False, "error": f"page_number must be one of {config.PAGES_TO_SCRAPE}"}
    url = _page_url(page_number)
    result = fetch(url)
    if not result["ok"]:
        STATE.failed[url] = result["error"]
        return {"ok": False, "page": page_number, "url": url, "error": result["error"]}
    return _ingest_listing(page_number, result["html"])


@tool
@logged
def scrape_product_page(product_url: str) -> dict:
    """Fetch one product page and store its category and stock text on the book's Bronze row."""
    if not is_valid_url(product_url):
        return {"ok": False, "error": f"invalid product url: {product_url}"}
    result = fetch(product_url)
    if not result["ok"]:
        STATE.failed[product_url] = result["error"]
        return {"ok": False, "product_url": product_url, "error": result["error"]}
    return _ingest_product(product_url, result["html"])


@tool
@logged
def scrape_product_pages(product_urls: list[str]) -> dict:
    """Fetch several product pages in one call (e.g. all 20 URLs from one listing page).
    Same as calling scrape_product_page for each URL; failed URLs are listed so you can retry them."""
    fetched, failed = 0, []
    for url in product_urls:
        if STOP_REQUESTED.is_set():
            break
        result = scrape_product_page.invoke({"product_url": url})  # logged per URL
        if result["ok"]:
            fetched += 1
        else:
            failed.append({"url": url, "error": result.get("error")})
    return {"ok": not failed, "fetched": fetched, "failed": failed,
            **({"error": f"{len(failed)} product pages failed"} if failed else {})}


@tool
@logged
def get_status() -> dict:
    """Progress report: pages done/remaining, books still missing category or stock, failed URLs, retry counts."""
    rows = db.bronze_rows(STATE.run_id)
    missing = sorted({r["product_url"] for r in rows if not r["category_raw"] or not r["stock_raw"]})
    return {
        "ok": True,
        "run_id": STATE.run_id,
        "pages_done": sorted(STATE.pages_done),
        "pages_remaining": [p for p in config.PAGES_TO_SCRAPE if p not in STATE.pages_done],
        "books_in_bronze": len({r["product_url"] for r in rows}),
        "books_missing_category_or_stock": len(missing),
        "missing_urls": missing,
        "failed_urls": STATE.failed,
        "retry_counts": STATE.retry_counts,
        "max_retries_per_url": config.MAX_RETRIES_PER_URL,
    }


@tool
@logged
def retry_failed(url: str, strategy: str) -> dict:
    """Retry a failed listing or product URL. strategy: "backoff" (wait 1s, 2s, 4s... then retry)
    or "alternate_user_agent" (retry immediately with a different User-Agent). Max 3 retries per URL."""
    if strategy not in ("backoff", "alternate_user_agent"):
        return {"ok": False, "error": 'strategy must be "backoff" or "alternate_user_agent"'}
    used = STATE.retry_counts.get(url, 0)
    if used >= config.MAX_RETRIES_PER_URL:
        log_event("retry_failed", "GAVE_UP", f"{url} reached {used} retries")
        return {"ok": False, "url": url, "error": f"max retries ({config.MAX_RETRIES_PER_URL}) reached; give up on this URL"}

    attempt = used + 1
    STATE.retry_counts[url] = attempt
    log_event("retry_failed", "RETRY", f"attempt {attempt}/{config.MAX_RETRIES_PER_URL} strategy={strategy} url={url}")

    user_agent = config.USER_AGENT
    if strategy == "backoff":
        time.sleep(2 ** (attempt - 1))
    else:
        user_agent = config.ALT_USER_AGENT

    result = fetch(url, user_agent=user_agent)
    if not result["ok"]:
        STATE.failed[url] = result["error"]
        return {"ok": False, "url": url, "attempt": attempt, "strategy": strategy, "error": result["error"]}

    # Re-run the same ingestion the original tool would have done
    page_match = re.search(r"page-(\d+)\.html$", url)
    if page_match:
        out = _ingest_listing(int(page_match.group(1)), result["html"])
    else:
        out = _ingest_product(url, result["html"])
    log_event("retry_failed", "RECOVERED", f"{url} recovered on attempt {attempt} using {strategy}")
    return {**out, "attempt": attempt, "strategy": strategy, "recovered": True}


def _clean_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Bronze rows -> typed rows. Returns (valid rows deduplicated by URL, counts of rejects)."""
    rejects = {"missing_price": 0, "bad_rating": 0, "missing_category": 0, "missing_stock": 0,
               "invalid_url": 0, "duplicates_removed": 0}
    by_url = {}
    for r in rows:
        clean = {
            "product_url": (r["product_url"] or "").strip(),
            "title": (r["title"] or "").strip(),
            "price_gbp": parse_price(r["price_raw"]),
            "rating": parse_rating(r["rating_raw"]),
            "category": (r["category_raw"] or "").strip() or None,
            "stock_qty": parse_stock(r["stock_raw"]),
        }
        if not is_valid_url(clean["product_url"]):
            rejects["invalid_url"] += 1
        elif clean["price_gbp"] is None:
            rejects["missing_price"] += 1
        elif clean["rating"] not in range(1, 6):
            rejects["bad_rating"] += 1
        elif clean["category"] is None:
            rejects["missing_category"] += 1
        elif clean["stock_qty"] is None:
            rejects["missing_stock"] += 1
        else:
            if clean["product_url"] in by_url:
                rejects["duplicates_removed"] += 1
            by_url[clean["product_url"]] = clean  # later scrape wins
    return list(by_url.values()), rejects


@tool
@logged
def validate_data() -> dict:
    """Check this run's Bronze data: missing prices, ratings outside 1-5, missing category/stock,
    invalid URLs and duplicate URLs. Returns a report with counts."""
    rows = db.bronze_rows(STATE.run_id)
    urls = [r["product_url"] for r in rows]
    ratings = [parse_rating(r["rating_raw"]) for r in rows]
    report = {
        "rows_checked": len(rows),
        "unique_urls": len(set(urls)),
        "missing_price": sum(parse_price(r["price_raw"]) is None for r in rows),
        "rating_outside_1_5": sum(x is None or not 1 <= x <= 5 for x in ratings),
        "missing_category": sum(not r["category_raw"] for r in rows),
        "missing_stock": sum(parse_stock(r["stock_raw"]) is None for r in rows),
        "invalid_urls": sum(not is_valid_url(u) for u in urls),
        "duplicate_rows": len(urls) - len(set(urls)),
    }
    issues = sum(v for k, v in report.items() if k not in ("rows_checked", "unique_urls"))
    return {"ok": True, "report": report, "total_issues": issues,
            "note": "Rows with issues are dropped and duplicates collapsed by save_clean_data."}


def _rebuild_gold(conn) -> None:
    """Recompute all Gold tables from Silver."""
    for table in ("gold_kpis", "gold_books_by_category", "gold_rating_distribution", "gold_top10_expensive"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute(
        """INSERT INTO gold_kpis
           SELECT ?, COUNT(*), ROUND(AVG(price_gbp), 2), MIN(price_gbp), MAX(price_gbp),
                  ROUND(AVG(rating), 2), SUM(stock_qty), ?
           FROM silver_books""",
        (STATE.run_id, db.now_iso()),
    )
    conn.execute(
        """INSERT INTO gold_books_by_category
           SELECT category, COUNT(*), ROUND(AVG(price_gbp), 2) FROM silver_books
           GROUP BY category ORDER BY COUNT(*) DESC, category"""
    )
    conn.execute(
        """INSERT INTO gold_rating_distribution
           SELECT rating, COUNT(*) FROM silver_books GROUP BY rating ORDER BY rating"""
    )
    conn.execute(
        """INSERT INTO gold_top10_expensive
           SELECT ROW_NUMBER() OVER (ORDER BY price_gbp DESC, title), title, price_gbp, rating, category, product_url
           FROM silver_books ORDER BY price_gbp DESC, title LIMIT 10"""
    )


@tool
@logged
def save_clean_data() -> dict:
    """Clean and type this run's Bronze rows, drop invalid and duplicate rows, upsert into Silver
    (no duplicates on rerun), then rebuild all Gold tables."""
    clean, rejects = _clean_rows(db.bronze_rows(STATE.run_id))
    if not clean:
        return {"ok": False, "error": "no valid rows to save; scrape first", "rejects": rejects}
    now = db.now_iso()
    with db.connect() as conn:
        conn.executemany(
            """INSERT INTO silver_books (product_url, title, price_gbp, rating, category, stock_qty, run_id, updated_at)
               VALUES (:product_url, :title, :price_gbp, :rating, :category, :stock_qty, :run_id, :updated_at)
               ON CONFLICT(product_url) DO UPDATE SET
                   title = excluded.title, price_gbp = excluded.price_gbp, rating = excluded.rating,
                   category = excluded.category, stock_qty = excluded.stock_qty,
                   run_id = excluded.run_id, updated_at = excluded.updated_at""",
            [{**row, "run_id": STATE.run_id, "updated_at": now} for row in clean],
        )
        _rebuild_gold(conn)
        silver_count = conn.execute("SELECT COUNT(*) FROM silver_books").fetchone()[0]
    return {"ok": True, "rows_upserted": len(clean), "rejected": rejects, "silver_row_count": silver_count}


@tool
@logged
def get_facts() -> dict:
    """Return the computed numbers from Gold: total books, average price, top categories,
    rating distribution and the most expensive book. The summary may only use these numbers."""
    kpis = db.query("SELECT * FROM gold_kpis")
    if not kpis:
        return {"ok": False, "error": "Gold tables are empty; run save_clean_data first"}
    k = kpis[0]
    top = db.query("SELECT * FROM gold_top10_expensive ORDER BY rank LIMIT 1")
    return {
        "ok": True,
        "total_books": k["total_books"],
        "average_price_gbp": k["avg_price"],
        "min_price_gbp": k["min_price"],
        "max_price_gbp": k["max_price"],
        "average_rating": k["avg_rating"],
        "total_stock": k["total_stock"],
        "top_categories": db.query(
            "SELECT category, book_count, avg_price FROM gold_books_by_category ORDER BY book_count DESC, category LIMIT 5"),
        "rating_distribution": {r["rating"]: r["book_count"] for r in
                                db.query("SELECT * FROM gold_rating_distribution ORDER BY rating")},
        "most_expensive_book": {"title": top[0]["title"], "price_gbp": top[0]["price_gbp"],
                                "category": top[0]["category"]} if top else None,
    }


@tool
@logged
def save_summary(text: str) -> dict:
    """Store the AI-written summary of the data (Gold). Use only numbers returned by get_facts."""
    if not text or not text.strip():
        return {"ok": False, "error": "summary text is empty"}
    with db.connect() as conn:
        conn.execute("DELETE FROM gold_summary")
        conn.execute("INSERT INTO gold_summary (run_id, created_at, summary) VALUES (?, ?, ?)",
                     (STATE.run_id, db.now_iso(), text.strip()))
    return {"ok": True, "chars": len(text.strip())}


@tool
@logged
def finish(reason: str) -> dict:
    """End the run. Call this once everything is done (or nothing more can be done), with a short reason."""
    STATE.finished = True
    STATE.finish_reason = reason
    return {"ok": True, "finished": True, "reason": reason}


TOOLS = [scrape_listing_page, scrape_product_page, scrape_product_pages, get_status, retry_failed, validate_data,
         save_clean_data, get_facts, save_summary, finish]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
