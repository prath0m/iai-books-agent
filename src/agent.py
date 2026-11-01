"""LangGraph agent: an LLM chooses every tool call; a tool node executes them and logs each one."""
import json
import threading
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src import config, tools

SYSTEM_PROMPT = f"""You are a data-pipeline agent. Your job is to scrape books from books.toscrape.com
into a SQLite Bronze -> Silver -> Gold pipeline and write a short factual summary. You do all work by calling tools.

Goal: all books from listing pages {config.PAGES_TO_SCRAPE} (20 books per page, 60 total), each with
title, price, rating, category and stock, saved to Silver/Gold, plus a summary.

How to work:
1. Plan first: briefly state your plan in one or two sentences.
2. Scrape all listing pages with scrape_listing_page. It returns the product URLs on that page.
3. Get category and stock for every product URL. Use scrape_product_pages and pass ALL the product
   URLs of a listing page (or all 60) in one call; scrape_product_page does a single URL.
   Be efficient: put independent tool calls in ONE turn (e.g. all three listing pages at once).
   Requests still run one after another, so this is polite.
4. If any tool returns ok=false, call get_status, then call retry_failed for the failed URL.
   Choose the strategy yourself: try "backoff" first for timeouts / 5xx errors, and if a retry fails,
   switch to the other strategy ("alternate_user_agent"). Never retry a URL more than
   {config.MAX_RETRIES_PER_URL} times; after that, give up on it and continue.
5. Before saving, call get_status to confirm nothing is missing, then call validate_data.
6. Call save_clean_data (it removes invalid and duplicate rows and builds the Gold tables).
7. Call get_facts, then write a 3-6 sentence summary and store it with save_summary.
   The summary may ONLY use numbers that appear in the get_facts result. Never invent or estimate numbers.
8. Call finish with a short reason (you may call it in the same turn as save_summary).

Before each tool call, write one short sentence saying why you are making it (this is logged).
Do not re-scrape pages or products that already succeeded."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _text_of(message: AIMessage) -> str:
    """Plain text of an AI message (Gemini may return a list of content blocks)."""
    if isinstance(message.content, str):
        return message.content.strip()
    parts = [b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text"]
    return " ".join(p.strip() for p in parts if p.strip())


def get_llm_with_tools():
    """Gemini with tools bound: the main model, then each fallback model if the previous one fails.

    The free tier allows only a few requests per minute per model, so each model gets its own rate
    limiter (calls are paced instead of failing with HTTP 429). Errors such as a used-up daily quota
    or a server timeout fall through to the next model."""
    bound = []
    for model in [config.MODEL] + config.FALLBACK_MODELS:
        limiter = InMemoryRateLimiter(requests_per_second=config.GOOGLE_REQUESTS_PER_MINUTE / 60,
                                      check_every_n_seconds=0.5, max_bucket_size=1)
        limiter.available_tokens = 1.0  # first call goes out immediately instead of waiting ~15s
        # No client retries: any error (quota used up, timeout) falls through to the next model at once
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_tokens=4096,
                                     timeout=45, max_retries=0, rate_limiter=limiter)
        bound.append(llm.bind_tools(tools.TOOLS))
    return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]


def build_graph(llm=None):
    """Graph: START -> agent -> (tools -> agent)* -> END. `llm` can be injected for testing."""
    llm_with_tools = get_llm_with_tools() if llm is None else llm.bind_tools(tools.TOOLS)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        served_by = response.response_metadata.get("model_name") or "model"
        thought = _text_of(response)
        calls = [c["name"] for c in response.tool_calls]
        plan = ", ".join(f"{n} x{calls.count(n)}" if calls.count(n) > 1 else n for n in dict.fromkeys(calls))
        console = f"\n[model: {served_by}]" + (f"\nAGENT: {thought}" if thought else "") + "".join(
            f"\n   -> decides to call {c['name']}({json.dumps(c['args'])})" for c in response.tool_calls)
        text = (thought or "Thinking") + (f"  ->  {plan}" if plan else "")
        tools.emit("agent", served_by, text, console)
        return {"messages": [response]}

    def tool_node(state: AgentState):
        """Run the requested tools sequentially (keeps requests polite and SQLite single-threaded)."""
        last: AIMessage = state["messages"][-1]
        tools.STATE.current_reason = _text_of(last)[:200]
        results = []
        for call in last.tool_calls:
            tool = tools.TOOLS_BY_NAME.get(call["name"])
            if tools.STOP_REQUESTED.is_set():
                output = {"ok": False, "error": "skipped: run stopped by user"}
            elif tool is None:
                output = {"ok": False, "error": f"unknown tool {call['name']}"}
            else:
                output = tool.invoke(call["args"])
            results.append(ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"]))
        tools.STATE.current_reason = ""
        return {"messages": results}

    def after_agent(state: AgentState):
        if tools.STOP_REQUESTED.is_set():
            return END
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AgentState):
        return END if tools.STATE.finished or tools.STOP_REQUESTED.is_set() else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


# Only one run at a time: tool state is shared, and a hosted dashboard can have many visitors
_RUN_LOCK = threading.Lock()


def run_agent(simulate_failure: bool = False, llm=None) -> dict:
    """Run one full pipeline. Returns {"run_id", "finished", "reason"}. Never raises on agent errors."""
    if not _RUN_LOCK.acquire(blocking=False):
        return {"run_id": None, "finished": False, "reason": "another pipeline run is already in progress"}
    try:
        return _run(simulate_failure, llm)
    finally:
        _RUN_LOCK.release()


def _run(simulate_failure: bool, llm) -> dict:
    run_id = f"run_{tools.db.now_iso().replace(':', '').replace('-', '')[:15]}_{uuid.uuid4().hex[:6]}"
    tools.start_run(run_id, simulate_failure=simulate_failure)
    tools.log_event("pipeline", "START", text=f"Run started (simulate failure {'on' if simulate_failure else 'off'})",
                    details=f"run_id={run_id} model={config.MODEL} fallbacks={config.FALLBACK_MODELS} simulate_failure={simulate_failure}")

    graph = build_graph(llm)
    try:
        graph.invoke(
            {"messages": [HumanMessage("Run the books pipeline now.")]},
            config={"recursion_limit": config.RECURSION_LIMIT},
        )
        if tools.STOP_REQUESTED.is_set():
            reason = "stopped by user"
            tools.log_event("pipeline", "STOPPED", "run stopped by user (Stop button)", text="Run stopped by user")
        else:
            reason = tools.STATE.finish_reason or "agent stopped without calling finish"
    except GraphRecursionError:
        reason = f"stopped: recursion limit {config.RECURSION_LIMIT} reached"
        tools.log_event("pipeline", "ERROR", reason)
    except Exception as exc:  # e.g. API key / network problems
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message:
            reason = ("stopped: Gemini free-tier quota is used up for every configured model "
                      "(about 20 requests/day per model). Try again later or add models to GOOGLE_FALLBACK_MODELS.")
        elif "API_KEY_INVALID" in message:
            reason = "stopped: GOOGLE_API_KEY is not valid. Check the key in .env (or the app's Secrets when hosted)."
        else:
            reason = f"stopped: {type(exc).__name__}: {message[:300]}"
        tools.log_event("pipeline", "ERROR", reason)

    tools.log_event("pipeline", "END", reason)
    return {"run_id": run_id, "finished": tools.STATE.finished, "reason": reason}
