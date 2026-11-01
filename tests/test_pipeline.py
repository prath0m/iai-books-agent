"""Offline tests: parsing and idempotent Silver upserts. Uses saved HTML snippets, no live calls."""
from pathlib import Path

import pytest

from src import config, db, tools

FIXTURES = Path(__file__).parent / "fixtures"
PAGE_URL = "https://books.toscrape.com/catalogue/page-1.html"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the pipeline at a throwaway SQLite file."""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    tools.start_run("test_run")
    return tmp_path / "test.db"


@pytest.mark.parametrize("css, expected", [
    ("star-rating One", 1), ("star-rating Three", 3), (["star-rating", "Five"], 5),
    ("star-rating Zero", None), ("", None), (None, None),
])
def test_parse_rating(css, expected):
    assert tools.parse_rating(css) == expected


@pytest.mark.parametrize("text, expected", [
    ("£51.77", 51.77), ("Â£53.74", 53.74), ("£10", 10.0), ("", None), ("n/a", None),
])
def test_parse_price(text, expected):
    assert tools.parse_price(text) == expected


def test_parse_stock():
    assert tools.parse_stock("In stock (22 available)") == 22
    assert tools.parse_stock("Out of stock") is None


def test_parse_listing_snippet():
    rows = tools.parse_listing((FIXTURES / "listing_snippet.html").read_bytes(), PAGE_URL)
    assert len(rows) == 3
    first = rows[0]
    assert first["title"] == "A Light in the Attic"
    assert tools.parse_price(first["price_raw"]) == 51.77
    assert tools.parse_rating(first["rating_raw"]) == 3
    assert first["product_url"] == "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html"


def test_parse_product_snippet():
    details = tools.parse_product((FIXTURES / "product_snippet.html").read_bytes())
    assert details["category_raw"] == "Poetry"
    assert tools.parse_stock(details["stock_raw"]) == 22


def test_save_twice_no_duplicates(temp_db):
    # Load Bronze from the saved snippets (same product page details for every book is fine here)
    rows = tools.parse_listing((FIXTURES / "listing_snippet.html").read_bytes(), PAGE_URL)
    details = tools.parse_product((FIXTURES / "product_snippet.html").read_bytes())
    db.insert_bronze_rows("test_run", PAGE_URL, rows)
    for r in rows:
        db.update_bronze_details("test_run", r["product_url"], details["category_raw"], details["stock_raw"])

    first = tools.save_clean_data.invoke({})
    count_after_first = db.query("SELECT COUNT(*) AS n FROM silver_books")[0]["n"]
    second = tools.save_clean_data.invoke({})
    count_after_second = db.query("SELECT COUNT(*) AS n FROM silver_books")[0]["n"]

    assert first["ok"] and second["ok"]
    assert count_after_first == count_after_second == 3
    # Gold is rebuilt, not appended
    assert db.query("SELECT total_books FROM gold_kpis") == [{"total_books": 3}]


def test_tools_never_raise(temp_db):
    # Bad input comes back as ok=False instead of an exception
    assert tools.scrape_listing_page.invoke({"page_number": 99})["ok"] is False
    assert tools.retry_failed.invoke({"url": PAGE_URL, "strategy": "magic"})["ok"] is False
    assert tools.get_facts.invoke({})["ok"] is False


def test_only_one_run_at_a_time():
    # A second click while a run is in progress must not start another run
    from src import agent
    with agent._RUN_LOCK:
        result = agent.run_agent()
    assert result["run_id"] is None and result["finished"] is False
