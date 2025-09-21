from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import logging
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import yfinance as yf
import logging
# import gradio as gr
from langchain_core.tools import tool
import json
from langchain_community.tools import BraveSearch
import os
from dotenv import load_dotenv
from tavily import TavilyClient
tavily_client = TavilyClient(api_key="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")

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


@tool
def get_stock_price(symbol: str) -> str:
    """Get compact current stock price and key valuation info."""
    logging.info(f"[TOOL] Fetching stock price for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        if hist.empty:
            return json.dumps({"error": f"Could not retrieve data for {symbol}"})

        current_price = round(hist['Close'].iloc[-1], 2)

        summary = {
            "symbol": symbol,
            "company": info.get('longName', symbol),
            "price": current_price,
            "market_cap_B": round(info.get('marketCap', 0) / 1e9, 2) if info.get('marketCap') else "N/A",
            "pe_ratio": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A",
            "52w_high": round(info.get('fiftyTwoWeekHigh', 0), 2),
            "52w_low": round(info.get('fiftyTwoWeekLow', 0), 2)
        }

        return json.dumps(summary, indent=2)
    except Exception as e:
        logging.exception("Exception in get_stock_price")
        return json.dumps({"error": str(e)})


@tool
def get_financial_statements(symbol: str) -> str:
    """Retrieve compact key financial statement data."""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        latest_year = financials.columns[0]

        revenue = float(financials.loc["Total Revenue", latest_year]) if "Total Revenue" in financials.index else 0
        net_income = float(financials.loc["Net Income", latest_year]) if "Net Income" in financials.index else 0
        assets = float(balance_sheet.loc["Total Assets", latest_year]) if "Total Assets" in balance_sheet.index else 0
        debt = float(balance_sheet.loc["Total Debt", latest_year]) if "Total Debt" in balance_sheet.index else 0

        summary = {
            "symbol": symbol,
            "year": str(latest_year.year),
            "revenue_B": round(revenue / 1e9, 2),
            "net_income_B": round(net_income / 1e9, 2),
            "assets_B": round(assets / 1e9, 2),
            "debt_B": round(debt / 1e9, 2)
        }

        return json.dumps(summary, indent=2)
    except Exception as e:
        logging.exception("Exception in get_financial_statements")
        return json.dumps({"error": str(e)})



brave_search = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
    search_kwargs={"count": 3}  
)


# @tool
# def search_financial_news(company_name: str, symbol: str) -> str:
#     """Search for recent financial news headlines about a company."""
#     try:
#         search_query = f"{company_name} {symbol} stock financial news latest"
#         results = brave_search.run(search_query)

#         # Extract just the top headlines (not full text blobs)
#         headlines = []
#         if isinstance(results, list):
#             for r in results[:5]:  # top 5
#                 if isinstance(r, dict) and "title" in r:
#                     headlines.append(r["title"])

#         summary = {
#             "symbol": symbol,
#             "company": company_name,
#             "headlines": headlines
#         }

#         return json.dumps(summary, indent=2)
#     except Exception as e:
#         logging.exception("Exception in search_financial_news")
#         return json.dumps({"error": str(e)})

@tool
def search_financial_news(company_name: str, symbol: str) -> str:
    """Search for recent financial news headlines about a company using Tavily Search."""
    try:
        query = f"{company_name} {symbol} stock financial news latest"
        results = tavily_client.search(query)

        # Tavily returns plain strings → keep top 5
        headlines = results[:5] if isinstance(results, list) else [str(results)]

        return json.dumps({
            "symbol": symbol,
            "company": company_name,
            "headlines": headlines
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search news: {str(e)}"})


# @tool
# def search_market_trends(topic: str) -> str:
#     """Search and summarize market trend headlines on a specific topic."""
#     try:
#         search_query = f"{topic} market trends 2024 2025 outlook forecast"
#         results = brave_search.run(search_query)

#         # Extract top 3–5 headlines only
#         headlines = []
#         if isinstance(results, list):
#             for r in results[:5]:
#                 if isinstance(r, dict) and "title" in r:
#                     headlines.append(r["title"])

#         summary = {
#             "topic": topic,
#             "headlines": headlines
#         }

#         return json.dumps(summary, indent=2)
#     except Exception as e:
#         logging.exception("Exception in search_market_trends")
#         return json.dumps({"error": str(e)})

@tool
def search_market_trends(topic: str) -> str:
    """Search for market trend headlines on a specific topic using Tavily Search."""
    try:
        query = f"{topic} market trends 2024 2025 outlook forecast"
        results = tavily_client.search(query)

        headlines = results[:5] if isinstance(results, list) else [str(results)]

        return json.dumps({
            "topic": topic,
            "headlines": headlines
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to search trends: {str(e)}"})


@tool
def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Calculate compact key technical indicators for a stock."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        if hist.empty:
            return json.dumps({"error": f"No historical data for {symbol}"})

        # Simple Moving Averages
        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()

        # RSI (14-day)
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest = hist.iloc[-1]

        summary = {
            "symbol": symbol,
            "price": round(latest["Close"], 2),
            "sma20": round(latest["SMA_20"], 2),
            "sma50": round(latest["SMA_50"], 2),
            "rsi": round(rsi.iloc[-1], 2),
            "trend": (
                "bullish"
                if latest["Close"] > latest["SMA_20"] > latest["SMA_50"]
                else "bearish"
            )
        }

        return json.dumps(summary, indent=2)
    except Exception as e:
        logging.exception("Exception in get_technical_indicators")
        return json.dumps({"error": str(e)})


# Sub-agent configurations
fundamental_analyst = {
    "name": "fundamental-analyst",
    "description": "Performs deep fundamental analysis of companies including financial ratios, growth metrics, and valuation",
    "prompt": """You are an expert fundamental analyst with 15+ years of experience. 
    Focus on:
    - Financial statement analysis
    - Ratio analysis (P/E, P/B, ROE, ROA, Debt-to-Equity)
    - Growth metrics and trends
    - Industry comparisons
    - Intrinsic value calculations
    Always provide specific numbers and cite your sources.""",
}

technical_analyst = {
    "name": "technical-analyst",
    "description": "Analyzes price patterns, technical indicators, and trading signals",
    "prompt": """You are a professional technical analyst specializing in chart analysis and trading signals.
    Focus on:
    - Price action and trend analysis
    - Technical indicators (RSI, MACD, Moving Averages)
    - Support and resistance levels
    - Volume analysis
    - Entry/exit recommendations
    Provide specific price levels and timeframes for your recommendations.""",
}

risk_analyst = {
    "name": "risk-analyst",
    "description": "Evaluates investment risks and provides risk assessment",
    "prompt": """You are a risk management specialist focused on identifying and quantifying investment risks.
    Focus on:
    - Market risk analysis
    - Company-specific risks
    - Sector and industry risks
    - Liquidity and credit risks
    - Regulatory and compliance risks
    Always quantify risks where possible and suggest mitigation strategies.""",
}

subagents = [fundamental_analyst, technical_analyst, risk_analyst]


# Main research instructions
research_instructions = """You are an elite stock research analyst with access to multiple specialized tools and sub-agents. 

Your research process should be systematic and comprehensive:

1. **Initial Data Gathering**: Start by collecting basic stock information, price data, and recent news
2. **News & Market Research**: Use Brave Search to find recent financial news and market trends
3. **Fundamental Analysis**: Deep dive into financial statements, ratios, and company fundamentals
4. **Technical Analysis**: Analyze price patterns, trends, and technical indicators
5. **Risk Assessment**: Identify and evaluate potential risks
6. **Competitive Analysis**: Compare with industry peers when relevant
7. **Synthesis**: Combine all findings into a coherent investment thesis
8. **Recommendation**: Provide clear buy/sell/hold recommendation with price targets

Always:
- Use specific data and numbers to support your analysis
- Cite your sources and methodology
- Include recent news and market sentiment in your analysis from Brave Search results
- Consider multiple perspectives and potential scenarios
- Provide actionable insights and concrete recommendations
- Structure your final report professionally

When using sub-agents, provide them with specific, focused tasks and incorporate their specialized insights into your overall analysis."""

research_instructions += """
IMPORTANT:
- Always output JSON with this exact structure:
{
  "todos": [
    {"name": "<string>", "status": "<pending|completed|error>"}
  ]
}
- Do NOT use keys like 'title' or 'state'.
- Do NOT include extra text outside JSON.
"""



_brave_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
tools = [get_stock_price, get_financial_statements, get_technical_indicators]
if _brave_api_key:
    tools.extend([search_financial_news, search_market_trends])
else:
    logging.info("BRAVE_SEARCH_API_KEY not found: excluding Brave Search tools from tools list")

# ---------- Define Schema ----------
class ResearchStep(BaseModel):
    name: str
    status: str  # "pending", "completed", "error"

class StockReport(BaseModel):
    todos: List[ResearchStep]
    fundamentals: Optional[Dict[str, str]] = None
    technicals: Optional[Dict[str, str]] = None
    risks: Optional[List[str]] = None
    news_summary: Optional[str] = None
    recommendation: Optional[str] = None


# ---------- Safe JSON Parser ----------
def safe_json_parse(raw: str) -> dict:
    """Try to safely parse JSON from raw string output."""
    try:
        return json.loads(raw)
    except Exception:
        # Try to strip markdown code fences
        fixed = raw.strip()
        if fixed.startswith("```json"):
            fixed = fixed[7:-3].strip()
        try:
            return json.loads(fixed)
        except Exception as e:
            logging.error(f"[safe_json_parse] Failed to parse JSON: {e}")
            return {"error": "Invalid JSON output", "raw": raw}


# ---------- Format Markdown for UI ----------
def format_report_markdown(report: StockReport) -> str:
    md = "## 📊 Stock Research Report\n\n"

    # Todo steps
    md += "### ✅ Research Steps\n"
    for step in report.todos:
        status_icon = "🟢" if step.status == "completed" else "🟡" if step.status == "pending" else "🔴"
        md += f"- {status_icon} **{step.name}** ({step.status})\n"

    # Fundamentals
    if report.fundamentals:
        md += "\n### 📑 Fundamentals\n"
        for k, v in report.fundamentals.items():
            md += f"- **{k}**: {v}\n"

    # Technicals
    if report.technicals:
        md += "\n### 📈 Technicals\n"
        for k, v in report.technicals.items():
            md += f"- **{k}**: {v}\n"

    # Risks
    if report.risks:
        md += "\n### ⚠️ Risks\n"
        for r in report.risks:
            md += f"- {r}\n"

    # News
    if report.news_summary:
        md += "\n### 📰 Recent News\n"
        md += f"{report.news_summary}\n"

    # Recommendation
    if report.recommendation:
        md += "\n### 🎯 Recommendation\n"
        md += f"**{report.recommendation}**\n"

    return md


# ---------- Main Function ----------
def run_stock_research(query: str, model_provider: str = "ollama"):
    """Run stock research and return structured JSON + Markdown for UI."""
    try:
        logging.info(f"[run_stock_research] Query: {query}")

        # ... (same model selection + create_deep_agent code here) ...
        if model_provider == "lm_studio":
            selected_model = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                model="local-model",
                temperature=0,
            )
        else:  # ollama
            selected_model = ChatOllama(
                model="mistral",
                temperature=0,
            )

        # Create agent with selected model
        agent = create_deep_agent(
            tools=tools,
            instructions=research_instructions,
            subagents=subagents,
            model=selected_model,
        )
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        raw_output = ""
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            raw_output = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")

        # Parse JSON safely
        parsed_json = safe_json_parse(raw_output)

        # Validate against schema (fall back to dict if invalid)
        try:
            report = StockReport(**parsed_json)
        except Exception:
            logging.warning("[run_stock_research] Output did not match schema, wrapping raw JSON")
            report = StockReport(todos=[ResearchStep(name="Unknown", status="error")])

        # Markdown for UI
        markdown_output = format_report_markdown(report)

        return {"json": parsed_json, "markdown": markdown_output}

    except Exception as e:
        logging.exception("[run_stock_research] Exception during invocation")
        return {"json": {"error": str(e)}, "markdown": f"❌ Error: {str(e)}"}


# with gr.Blocks() as demo:
#     gr.Markdown("## 📊 Stock Research Agent with Brave Search")

#     with gr.Row():
#         model_dropdown = gr.Dropdown(
#             choices=["ollama", "lm_studio"],
#             value="ollama",
#             label="Model Provider",
#         )

#     with gr.Row():
#         query_input = gr.Textbox(label="Research Query", lines=6)

#     run_button = gr.Button("Run Analysis")
#     output_box = gr.Textbox(label="Research Report", lines=20)

#     output_md = gr.Markdown()

#     def run_and_format(query, model_provider):
#         result = run_stock_research(query, model_provider)
#         return result["markdown"]

#     run_button.click(fn=run_and_format, inputs=[query_input, model_dropdown], outputs=output_box)

# demo.launch()


# import logging
# import json
# import re

# def extract_json(raw: str) -> str:
#     """Extract first JSON object/array from text output."""
#     match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
#     if match:
#         return match.group(0)
#     return raw

# def safe_json_parse(raw: str):
#     """Try to parse JSON safely, repair if needed."""
#     try:
#         return json.loads(raw)
#     except Exception:
#         fixed = extract_json(raw)
#         try:
#             return json.loads(fixed)
#         except Exception as e:
#             logging.error(f"[safe_json_parse] Failed to parse JSON: {e}")
#             return {"error": "Invalid JSON output", "raw": raw}

# def run_debug(query: str, model_provider="ollama"):
#     from research_agent import run_stock_research  # import your function

#     print("\n=== USER QUERY ===")
#     print(query)

#     # Run agent
#     result = run_stock_research(query, model_provider)

#     print("\n=== RAW OUTPUT FROM MODEL ===")
#     print(result)

#     # Try parsing JSON
#     parsed = safe_json_parse(result)
#     print("\n=== PARSED JSON ===")
#     print(json.dumps(parsed, indent=2))

#     return parsed

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)

#     query = "Comprehensive 3-month analysis of Apple Inc. (AAPL)"
#     run_debug(query)
