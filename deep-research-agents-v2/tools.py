
from langchain_core.tools import tool
import json
import yfinance as yf
import logging
from langchain_community.tools import BraveSearch
from tavily import TavilyClient
import os

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
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
