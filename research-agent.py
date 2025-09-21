from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import logging
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
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information."""
    logging.info(f"[TOOL] Fetching stock price for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1mo")
        if hist.empty:
            logging.error("No historical data found")
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
        logging.info(f"[TOOL RESULT] {result}")
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
        return f"Error: {str(e)}"


brave_search = BraveSearch.from_api_key(
    api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
    search_kwargs={"count": 3}  
)

@tool
def search_financial_news(company_name: str, symbol: str) -> str:
    """Search for recent financial news about a company using Brave Search."""
    try:
        search_query = f"{company_name} {symbol} financial news stock earnings latest"
        results = brave_search.run(search_query)
        
        return json.dumps({
            "symbol": symbol,
            "company": company_name,
            "search_query": search_query,
            "news_results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to search news: {str(e)}"})


@tool
def search_market_trends(topic: str) -> str:
    """Search for market trends and analysis on a specific topic using Brave Search."""
    try:
        search_query = f"{topic} market analysis trends 2024 2025 investment outlook forecast"
        results = brave_search.run(search_query)
        
        return json.dumps({
            "topic": topic,
            "search_query": search_query,
            "trend_results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to search trends: {str(e)}"})


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
        return f"Error: {str(e)}"


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
                model="gpt-oss",
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


with gr.Blocks() as demo:
    gr.Markdown("## 📊 Stock Research Agent with Brave Search")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["ollama", "lm_studio"],
            value="ollama",
            label="Model Provider",
        )

    with gr.Row():
        query_input = gr.Textbox(label="Research Query", lines=6)

    run_button = gr.Button("Run Analysis")
    output_box = gr.Textbox(label="Research Report", lines=20)

    output_md = gr.Markdown()

    def run_and_format(query, model_provider):
        result = run_stock_research(query, model_provider)
        return result["markdown"]

    run_button.click(fn=run_and_format, inputs=[query_input, model_dropdown], outputs=output_box)

demo.launch()
