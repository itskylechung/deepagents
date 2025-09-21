from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import yfinance as yf
import logging
import gradio as gr
from langchain_core.tools import tool
import json
from langchain_community.tools import BraveSearch
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from tools import *
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# === Load Environment ===
load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20B")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
DEFAULT_MODEL_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "ollama")
RECURSION_LIMIT = os.getenv("RECURSION_LIMIT", 25)

# Default: no search
web_search = None

if BRAVE_SEARCH_API_KEY:
    brave_search = BraveSearch.from_api_key(
        api_key=BRAVE_SEARCH_API_KEY,
        search_kwargs={"count": 3}
    )

    def web_search_func(query: str):
        return brave_search.run(query)
    
    web_search = web_search_func
    SEARCH_PROVIDER = "brave"

    logging.info("✅ Using Brave Search as web_search")

elif TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

    def web_search_func(query: str, max_results: int = 5):
        return tavily_client.search(query, max_results=max_results)

    web_search = web_search_func
    SEARCH_PROVIDER = "tavily"

    logging.info("✅ Using Tavily Search as web_search")

else:
    logging.warning("⚠️ No search provider configured. web_search will be None")








with open("instructions.md", "r") as f:
    CORE_INSTRUCTIONS = f.read()



with open("subagents.json", "r") as f:
    subagents_config = json.load(f)

fundamental_analyst = subagents_config["fundamental_analyst"]
technical_analyst = subagents_config["technical_analyst"]
risk_analyst = subagents_config["risk_analyst"]

def get_search_status() -> str:
    """Return a Markdown-formatted status message for web search setup."""
    status_msg = []

    # Brave
    
    if BRAVE_SEARCH_API_KEY:
        status_msg.append("### 🔎 Brave Search API Setup:\n")
        status_msg.append(f"**Current Status:** ✅ API Key detected\n\n")
    
    # Tavily

    elif TAVILY_API_KEY:
        status_msg.append("### 🌐 Tavily Search API Setup:\n")
        status_msg.append(f"**Current Status:** ✅ API Key detected\n\n")
    else:
        status_msg.append(f"**Current Status:** ❌ No API Key in `.env`\n\n")
        status_msg.append("- Get your free API key at [Brave Search API](https://api.search.brave.com/)\n")
        status_msg.append("- Get your free API key at [Tavily](https://tavily.com/)\n")
        status_msg.append("- Add it to your `.env` file as:\n")
        status_msg.append("  ```env\n  BRAVE_SEARCH_API_KEY=your_api_key_here\n  ```\n")

    return "".join(status_msg)


# === Tools ===
@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information."""
    logging.info(f"[TOOL] Fetching stock price for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        if hist.empty:
            return json.dumps({"error": f"Could not retrieve data for {symbol}"})

        current_price = hist['Close'].iloc[-1]
        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "company_name": info.get('longName', symbol),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0)
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        logging.exception("Exception in get_stock_price")
        return json.dumps({"error": str(e)})


@tool
def get_financial_statements(symbol: str) -> str:
    """Retrieve key financial statement data."""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        latest_year = financials.columns[0]

        return json.dumps(
            {
                "symbol": symbol,
                "period": str(latest_year.year),
                "revenue": (
                    float(financials.loc["Total Revenue", latest_year])
                    if "Total Revenue" in financials.index
                    else "N/A"
                ),
                "net_income": (
                    float(financials.loc["Net Income", latest_year])
                    if "Net Income" in financials.index
                    else "N/A"
                ),
                "total_assets": (
                    float(balance_sheet.loc["Total Assets", latest_year])
                    if "Total Assets" in balance_sheet.index
                    else "N/A"
                ),
                "total_debt": (
                    float(balance_sheet.loc["Total Debt", latest_year])
                    if "Total Debt" in balance_sheet.index
                    else "N/A"
                ),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


brave_search = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
    search_kwargs={"count": 3}  
)

# @tool
# def search_financial_news(company_name: str, symbol: str) -> str:
#     """Search for recent financial news about a company using Tavily."""
#     try:
#         search_query = f"{company_name} {symbol} financial news stock earnings latest"
#         results = tavily_client.search(search_query, max_results=5)
#         return json.dumps({
#             "symbol": symbol,
#             "company": company_name,
#             "search_query": search_query,
#             "news_results": results
#         }, indent=2)
#     except Exception as e:
#         return json.dumps({"error": f"Failed to search news: {str(e)}"})


# @tool
# def search_market_trends(topic: str) -> str:
#     """Search for market trends and analysis on a specific topic."""
#     try:
#         search_query = f"{topic} market analysis trends 2024 2025 investment outlook forecast"
#         results = tavily_client.search(search_query, max_results=5)
#         return json.dumps({
#             "topic": topic,
#             "search_query": search_query,
#             "trend_results": results
#         }, indent=2)
#     except Exception as e:
#         return json.dumps({"error": f"Failed to search trends: {str(e)}"})



@tool
def search_market_trends(topic: str) -> str:
    """Search for market trends and analysis on a specific topic using Brave or Tavily Search."""
    if not web_search:
        return json.dumps({"error": "No search provider configured"})

    try:
        search_query = f"{topic} market analysis trends 2024 2025 investment outlook forecast"
        results = web_search(search_query)   # 👈 unified function

        return json.dumps({
            "topic": topic,
            "search_query": search_query,
            "trend_results": results
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to search trends: {str(e)}"})
    
@tool
def search_financial_news(company_name: str, symbol: str) -> str:
    """Search for recent financial news about a company using Brave/ Tavily Search.
    Call this tool ONLY ONCE per query, unless specifically asked for additional news.
    If news results are already available, do not call again."""
    if not web_search:
        return json.dumps({"error": "No search provider configured"})

    try:
        query = f"{company_name} {symbol} financial news stock earnings latest"
        results = web_search(query)   # 👈 unified call
        return json.dumps({
            "symbol": symbol,
            "company": company_name,
            "results": results
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Calculate key technical indicators."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return json.dumps({"error": f"No historical data for {symbol}"})

        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()

        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest = hist.iloc[-1]
        latest_rsi = rsi.iloc[-1]

        return json.dumps(
            {
                "symbol": symbol,
                "current_price": round(latest["Close"], 2),
                "sma_20": round(latest["SMA_20"], 2),
                "sma_50": round(latest["SMA_50"], 2),
                "rsi": round(latest_rsi, 2),
                "volume": int(latest["Volume"]),
                "trend_signal": (
                    "bullish"
                    if latest["Close"] > latest["SMA_20"] > latest["SMA_50"]
                    else "bearish"
                ),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


# === Sub-Agents ===
# fundamental_analyst = {
#     "name": "fundamental-analyst",
#     "description": "Performs company fundamental analysis",
#     "prompt": """You are a fundamental analyst. Focus only on:
#     - Financial statements (Revenue, Net Income, Assets, Debt)
#     - Ratios: P/E, P/B, ROE, ROA, Debt/Equity
#     - Growth trends vs peers
#     - Valuation (intrinsic value)"""
# }

# technical_analyst = {
#     "name": "technical-analyst",
#     "description": "Analyzes technical signals",
#     "prompt": """You are a technical analyst. Focus only on:
#     - Price trends and patterns
#     - Indicators: RSI, MACD, MA, Bollinger Bands
#     - Support/resistance levels
#     - Short-term entry/exit signals"""
# }

# risk_analyst = {
#     "name": "risk-analyst",
#     "description": "Assesses investment risks",
#     "prompt": """You are a risk analyst. Focus only on:
#     - Market/systemic risks
#     - Company-specific risks
#     - Sector/industry risks
#     - Credit/liquidity/regulatory factors
#     - Mitigation strategies"""
# }



core_instructions = """You are a stock research agent with tools + sub-agents.

# Available Tools
- get_stock_price(symbol): Fetch current stock price, company name, market cap, P/E ratio, and 52-week range.
- get_financial_statements(symbol): Retrieve revenue, net income, assets, and debt from the latest financial statements.
- get_technical_indicators(symbol, period="3mo"): Calculate SMA (20/50), RSI, volume, and generate trend signals (bullish/bearish).
- search_financial_news(company_name, symbol): Search recent company-specific financial news, earnings, and market updates.
- search_market_trends(topic): Search broader market/sector trends and investment outlook.

# Output Rules
- Tool calls → ONLY JSON (no markdown, no text).
- Final report → Markdown with sections (Financials, Technicals, Risks, Recommendation).
- Do not mix JSON + Markdown in one response.

# Workflow
1. Gather stock basics & price
2. Get recent company news (once)
3. Fundamental analysis
4. Technical analysis
5. Risk assessment
6. Compare with peers if relevant
7. Synthesize into investment thesis
8. Conclude with Buy/Sell/Hold + price targets

# Stock Research Report – {company} ({symbol})

## 1. Company Snapshot
...

## 2. Recent News
...

## 3. Fundamentals
...

## 4. Technicals
...

## 5. Risks
...

## 6. Competitive Landscape
...

## 7. Investment Thesis
...

## 8. Recommendation
**Verdict:** Buy/Sell/Hold  
**Target Price:** $XXX – $YYY (3-month horizon)"""

tools = [get_stock_price, get_financial_statements, get_technical_indicators]
if web_search:
    tools.extend([search_financial_news, search_market_trends])
else:
    logging.info("⚠️ No web search provider configured (Brave/Tavily). Excluding search tools.")

# tools = [get_stock_price, get_financial_statements, get_technical_indicators, search_financial_news, search_market_trends]
subagents = [fundamental_analyst, technical_analyst, risk_analyst]


# === Runner ===
def run_stock_research(query: str, model_provider: str = DEFAULT_MODEL_PROVIDER):
    """Run the stock research agent and return the final message content with debug logging."""
    try:
        logging.info(f"[run_stock_research] Query received: {query}")
        logging.info(f"[run_stock_research] Model provider: {model_provider}")

        # Create model based on selection
        if model_provider == "lm_studio":
            selected_model = ChatOpenAI(
                base_url=LM_STUDIO_BASE_URL,
                api_key=LM_STUDIO_API_KEY,
                model=LM_STUDIO_MODEL,
                temperature=0,
            )
        else:  # ollama
            selected_model = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
            )

        # Create agent with selected model
        agent = create_deep_agent(
            tools=tools,
            instructions=CORE_INSTRUCTIONS,
            subagents=subagents,
            model=selected_model,
        ).with_config({"recursion_limit": int(RECURSION_LIMIT)})

        logging.debug(f"[run_stock_research] Research Instructions:\n" + core_instructions)
        logging.debug(f"[run_stock_research] Subagents:\n" + json.dumps(subagents, indent=2))

        print(query,"query")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]},{"recursion_limit":30})

        logging.debug(f"[run_stock_research] Full result: {result}")

        messages = result.get("messages", [])
        output_text = ""

        if not messages:
            logging.warning("[run_stock_research] No messages returned in result.")
            output_text = "Error: No response messages received."
        elif isinstance(messages[-1], dict):
            output_text = messages[-1].get("content", "")
            logging.debug(f"[run_stock_research] Output content from dict: {output_text}")
        elif hasattr(messages[-1], "content"):
            output_text = messages[-1].content
            logging.debug(f"[run_stock_research] Output content from object: {output_text}")
        else:
            logging.error("[run_stock_research] Unrecognized message format.")
            output_text = "Error: Invalid response message format."

        file_output = ""
        if "files" in result:
            file_output += "\n\n=== Generated Research Files ===\n"
            for filename, content in result["files"].items():
                preview = content[:500] + "..." if len(content) > 500 else content
                file_output += f"\n**{filename}**\n{preview}\n"
                logging.debug(f"[run_stock_research] File: {filename}, Preview: {preview[:100]}")

        return output_text + file_output
    except Exception as e:
        logging.exception("[run_stock_research] Exception during invocation:")
        return f"Error: {str(e)}"

# === Gradio App ===
with gr.Blocks() as demo:
    # gr.Markdown("## 📊 Stock Research Agent with Brave Search")
    gr.Markdown("## 📊 Stock Research Agent")

    gr.Markdown("Enter your stock research request below. Example: *Comprehensive analysis on Apple Inc. (AAPL)*")
    gr.Markdown(get_search_status())   # ✅ API setup info shown here

    # Check if API key is loaded from .env
    # env_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    # api_status = "✅ API Key loaded from .env" if env_api_key else "❌ No API Key in .env"
    
    # gr.Markdown(f"""
    # **Brave Search API Setup:**
    # 1. Get your free API key at [Brave Search API](https://api.search.brave.com/)
    # 2. Create a `.env` file in this directory with: `BRAVE_SEARCH_API_KEY=your_api_key_here`
    
    # **Current Status:** {api_status}
    # {f"**Loaded Key:** {env_api_key[:8]}...{env_api_key[-4:]} (masked)" if env_api_key else ""}
    # """)

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["ollama", "lm_studio"],
            value="ollama",
            label="Model Provider",
            info="Choose between Ollama (local) or LM Studio (local)",
        )

    with gr.Row():
        query_input = gr.Textbox(label="Research Query", lines=6, placeholder="Type your research query here...")

    run_button = gr.Button("Run Analysis")
    # output_box = gr.Textbox(label="Research Report", lines=20)
    output_box = gr.Markdown()
    run_button.click(fn=run_stock_research, inputs=[query_input, model_dropdown], outputs=output_box)

# Launch app
demo.launch(server_name="0.0.0.0", server_port=8001)
