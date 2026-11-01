"""Central configuration. Values come from environment variables / .env (never hard-coded secrets)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = folder that contains src/
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- LLM ---
# Google Gemini drives the agent (free tier). Key: GOOGLE_API_KEY in .env
MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
GOOGLE_REQUESTS_PER_MINUTE = float(os.getenv("GOOGLE_REQUESTS_PER_MINUTE", "4"))  # free tier limit is 5/min

# --- Storage ---
DB_PATH = ROOT_DIR / "data" / "books.db"

# --- Scraping ---
BASE_URL = "https://books.toscrape.com/catalogue/"
LISTING_URL = BASE_URL + "page-{page}.html"
PAGES_TO_SCRAPE = [1, 2, 3]
REQUEST_DELAY_SECONDS = 0.3  # politeness delay before every request
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "iai-books-agent/1.0 (educational demo scraper; contact: see README)"
ALT_USER_AGENT = "Mozilla/5.0 (compatible; iai-books-agent-retry/1.0)"

# --- Agent safety limits ---
MAX_RETRIES_PER_URL = 3
RECURSION_LIMIT = 200  # max LangGraph steps (one agent turn + one tool turn = 2 steps)


def simulate_failure_from_env() -> bool:
    """True if SIMULATE_FAILURE env var is set to a truthy value."""
    return os.getenv("SIMULATE_FAILURE", "0").strip().lower() in ("1", "true", "yes", "on")
