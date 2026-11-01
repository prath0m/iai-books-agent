"""LangGraph agent: an LLM chooses every tool call; a tool node executes them and logs each one."""
import json
import uuid
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
2. Scrape each listing page with scrape_listing_page. It returns the product URLs on that page.
3. Scrape every product URL with scrape_product_page to get category and stock. You may request many
   product pages in a single turn (e.g. all 20 URLs of a page at once) - they run one after another.
4. If any tool returns ok=false, call get_status, then call retry_failed for the failed URL.
   Choose the strategy yourself: try "backoff" first for timeouts / 5xx errors, and if a retry fails,
   switch to the other strategy ("alternate_user_agent"). Never retry a URL more than
   {config.MAX_RETRIES_PER_URL} times; after that, give up on it and continue.
5. Before saving, call get_status to confirm nothing is missing, then call validate_data.
6. Call save_clean_data (it removes invalid and duplicate rows and builds the Gold tables).
7. Call get_facts, then write a 3-6 sentence summary and store it with save_summary.
   The summary may ONLY use numbers that appear in the get_facts result. Never invent or estimate numbers.
8. Call finish with a short reason.

Before each tool call, write one short sentence saying why you are making it (this is logged).
Do not re-scrape pages or products that already succeeded."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _text_of(message: AIMessage) -> str:
    """Plain text of an AI message (Anthropic returns a list of content blocks)."""
    if isinstance(message.content, str):
        return message.content.strip()
    parts = [b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text"]
    return " ".join(p.strip() for p in parts if p.strip())


def get_llm():
    """Chat model for the configured provider (LLM_PROVIDER in .env)."""
    if config.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # imported lazily: optional provider
        return ChatGoogleGenerativeAI(model=config.MODEL, temperature=0, max_tokens=4096)
    return ChatAnthropic(model=config.MODEL, temperature=0, max_tokens=4096)


def build_graph(llm=None):
    """Graph: START -> agent -> (tools -> agent)* -> END. `llm` can be injected for testing."""
    if llm is None:
        llm = get_llm()
    llm_with_tools = llm.bind_tools(tools.TOOLS)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        thought = _text_of(response)
        if thought:
            print(f"\n🤖 AGENT: {thought}", flush=True)
        for call in response.tool_calls:
            print(f"   -> decides to call {call['name']}({json.dumps(call['args'])})", flush=True)
        return {"messages": [response]}

    def tool_node(state: AgentState):
        """Run the requested tools sequentially (keeps requests polite and SQLite single-threaded)."""
        last: AIMessage = state["messages"][-1]
        tools.STATE.current_reason = _text_of(last)[:200]
        results = []
        for call in last.tool_calls:
            tool = tools.TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                output = {"ok": False, "error": f"unknown tool {call['name']}"}
            else:
                output = tool.invoke(call["args"])
            results.append(ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"]))
        tools.STATE.current_reason = ""
        return {"messages": results}

    def after_agent(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AgentState):
        return END if tools.STATE.finished else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def run_agent(simulate_failure: bool = False, llm=None) -> dict:
    """Run one full pipeline. Returns {"run_id", "finished", "reason"}. Never raises on agent errors."""
    run_id = f"run_{tools.db.now_iso().replace(':', '').replace('-', '')[:15]}_{uuid.uuid4().hex[:6]}"
    tools.start_run(run_id, simulate_failure=simulate_failure)
    tools.log_event("pipeline", "START", f"run_id={run_id} provider={config.LLM_PROVIDER} model={config.MODEL} simulate_failure={simulate_failure}")

    graph = build_graph(llm)
    try:
        graph.invoke(
            {"messages": [HumanMessage("Run the books pipeline now.")]},
            config={"recursion_limit": config.RECURSION_LIMIT},
        )
        reason = tools.STATE.finish_reason or "agent stopped without calling finish"
    except GraphRecursionError:
        reason = f"stopped: recursion limit {config.RECURSION_LIMIT} reached"
        tools.log_event("pipeline", "ERROR", reason)
    except Exception as exc:  # e.g. API key / network problems
        reason = f"stopped: {type(exc).__name__}: {exc}"
        tools.log_event("pipeline", "ERROR", reason)

    tools.log_event("pipeline", "END", reason)
    return {"run_id": run_id, "finished": tools.STATE.finished, "reason": reason}
