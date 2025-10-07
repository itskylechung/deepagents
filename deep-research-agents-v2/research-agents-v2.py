from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import yfinance as yf
import logging
import gradio as gr
from langchain_core.tools import tool
import json
import os
from dotenv import load_dotenv
from tools import *



logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# === Load Environment ===
load_dotenv()


## === Load Model Configs ===
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-5-mini")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "gpt-4o-mini")
RECURSION_LIMIT = os.getenv("RECURSION_LIMIT", 25)
GRADIO_SERVER_NAME =os.getenv("GRADIO_SERVER_NAME","")
GRADIO_SERVER_PORT = os.getenv("GRADIO_SERVER_PORT","")

## === Load Instructions ===
with open("instructions.md", "r") as f:
    CORE_INSTRUCTIONS = f.read()

## === Load Sub Agents ===
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
        if model_provider == "open_ai":
            selected_model = ChatOpenAI(
                api_key=OPEN_AI_API_KEY,
                model=OPEN_AI_MODEL,
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
demo.launch(server_name=GRADIO_SERVER_NAME, server_port=int(GRADIO_SERVER_PORT))
