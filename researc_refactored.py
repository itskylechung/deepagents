from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import BraveSearch
import yfinance as yf
import logging
import gradio as gr
import json
import os
from tavily import TavilyClient
from config import Config


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== TOOLS ==========

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        if hist.empty:
            return json.dumps({"error": f"Could not retrieve data for {symbol}"})

        current_price = hist['Close'].iloc[-1]
        return json.dumps({
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "company_name": info.get('longName', symbol),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0)
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_financial_statements(symbol: str) -> str:
    """Retrieve key financial statement data."""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        latest_year = financials.columns[0]

        return json.dumps({
            "symbol": symbol,
            "period": str(latest_year.year),
            "revenue": float(financials.loc["Total Revenue", latest_year]) if "Total Revenue" in financials.index else "N/A",
            "net_income": float(financials.loc["Net Income", latest_year]) if "Net Income" in financials.index else "N/A",
            "total_assets": float(balance_sheet.loc["Total Assets", latest_year]) if "Total Assets" in balance_sheet.index else "N/A",
            "total_debt": float(balance_sheet.loc["Total Debt", latest_year]) if "Total Debt" in balance_sheet.index else "N/A"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


web_search = None
if Config.BRAVE_SEARCH_API_KEY:
    brave_search = BraveSearch.from_api_key(
        api_key=Config.BRAVE_SEARCH_API_KEY,
        search_kwargs={"count": 3}
    )

    def web_search_func(query: str):
        return brave_search.run(query)

    web_search = web_search_func
    logging.info("✅ Using Brave Search as web_search")

elif Config.TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=Config.TAVILY_API_KEY)

    def web_search_func(query: str):
        return tavily_client.search(query)

    web_search = web_search_func
    logging.info("✅ Using Tavily Search as web_search")

else:
    logging.warning("⚠️ No search provider configured. web_search will be None")


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
            return f"Error: No historical data for {symbol}"

        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()

        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest = hist.iloc[-1]
        return json.dumps({
            "symbol": symbol,
            "current_price": round(latest["Close"], 2),
            "sma_20": round(latest["SMA_20"], 2),
            "sma_50": round(latest["SMA_50"], 2),
            "rsi": round(rsi.iloc[-1], 2),
            "volume": int(latest["Volume"]),
            "trend_signal": "bullish" if latest["Close"] > latest["SMA_20"] > latest["SMA_50"] else "bearish"
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

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
    


# ========== SUBAGENTS ==========
# Sub-agent configurations
# fundamental_analyst = {
#         "name": "fundamental-analyst",
#         "description": "Performs deep fundamental analysis of companies including financial ratios, growth metrics, and valuation",
#         "prompt": """You are an expert fundamental analyst with 15+ years of experience. 
#         Focus on:
#         - Financial statement analysis
#         - Ratio analysis (P/E, P/B, ROE, ROA, Debt-to-Equity)
#         - Growth metrics and trends
#         - Industry comparisons
#         - Intrinsic value calculations
#         Always provide specific numbers and cite your sources.""",
#     }
    
# technical_analyst = {
#         "name": "technical-analyst",
#         "description": "Analyzes price patterns, technical indicators, and trading signals",
#         "prompt": """You are a professional technical analyst specializing in chart analysis and trading signals.
#         Focus on:
#         - Price action and trend analysis
#         - Technical indicators (RSI, MACD, Moving Averages)
#         - Support and resistance levels
#         - Volume analysis
#         - Entry/exit recommendations
#         Provide specific price levels and timeframes for your recommendations.""",
#     }
    
# risk_analyst = {
#         "name": "risk-analyst",
#         "description": "Evaluates investment risks and provides risk assessment",
#         "prompt": """You are a risk management specialist focused on identifying and quantifying investment risks.
#         Focus on:
#         - Market risk analysis
#         - Company-specific risks
#         - Sector and industry risks
#         - Liquidity and credit risks
#         - Regulatory and compliance risks
#         Always quantify risks where possible and suggest mitigation strategies.""",
#     }
# subagents = [fundamental_analyst, technical_analyst, risk_analyst]
subagents = Config.SUBAGENTS

tools = [get_stock_price, get_financial_statements, get_technical_indicators]
if web_search:
    tools.extend([search_financial_news, search_market_trends])
else:
    logging.info("⚠️ No web search provider configured (Brave/Tavily). Excluding search tools.")


# ========== MAIN RUNNER ==========
def run_stock_research(query: str, model_provider: str = Config.MODEL_PROVIDER):
    try:
        if model_provider == "lm_studio":
            model = ChatOpenAI(
                base_url=Config.LM_STUDIO_BASE_URL,
                api_key=Config.LM_STUDIO_API_KEY,
                model=Config.LM_STUDIO_MODEL,
                temperature=0,
            )
        else:
            model = ChatOllama(
                model=Config.OLLAMA_MODEL,
                temperature=0,
            )

        agent = create_deep_agent(
            tools=tools,
            instructions=Config.RESEARCH_INSTRUCTIONS,
            subagents=subagents,
            model=model,
        )
        logging.debug(f"[run_stock_research] Research Instructions:\n" + Config.RESEARCH_INSTRUCTIONS)
        logging.debug(f"[run_stock_research] Subagents:\n" + json.dumps(subagents, indent=2))
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

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
        logging.exception("[run_stock_research] Exception")
        return f"Error: {str(e)}"


# ========== RUN SCRIPT ==========
if __name__ == "__main__":
    query = "Conduct a comprehensive analysis of Apple Inc. (AAPL) for a 3-month investment horizon."
    output = run_stock_research(query)
    print(output)

    # Launch Gradio (optional)
    # with gr.Blocks() as demo:
    #     query_input = gr.Textbox(label="Research Query")
    #     output_box = gr.Markdown()
    #     run_button = gr.Button("Run Analysis")
    #     run_button.click(fn=run_stock_research, inputs=query_input, outputs=output_box)
    # demo.launch(server_name=Config.GRADIO_SERVER_NAME, server_port=Config.GRADIO_SERVER_PORT)