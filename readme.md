# 📊 Stock Research Agent – Version 2


* [Version 2](./deep-research-agents-v2)



## 🚀 Overview

This is the **second version** of the Stock Research Agent built with Gradio, LangChain DeepAgents, and optional Brave/Tavily integration. It improves significantly over the first version by applying **context engineering**, simplifying configuration, and optimizing tool usage.

---

## 📊 Version Comparison (v1 vs v2)

| Feature                   | Version 1 (Old)        | Version 2 (New)                             |
| ------------------------- | ---------------------- | ------------------------------------------- |
| **Configuration**         | Hardcoded in `.py`     | `.env`, `.json`, `.md` external configs     |
| **Search Provider**       | Fixed to Brave         | Optional Brave/Tavily, auto-selected        |
| **Subagents**             | Defined inline in code | Externalized in `subagents.json`            |
| **Core Instructions**     | Inline (long)          | Moved to `instructions.md`                  |
| **Token Usage per Query** | \~13,000               | \~3,500                                     |
| **Response Time**         | Slow (bloated context) | 60–70% faster (context engineered)          |
| **UI**                    | Basic textbox          | Markdown + Chat UI                          |
| **Error Handling**        | Minimal                | Graceful fallbacks, provider status banners |

---

## 📉 Token Usage Optimization

* **Before (v1):** \~13,000 tokens per research query (bloated instructions + redundant context).
* **Now (v2):** \~3,500 tokens per query.
* Achieved through:

  * Externalizing prompts (`.md` + `.json` files).
  * Removing redundant instructions.
  * Adding clear tool descriptions.
  * Context-aware selective tool usage.

💰 **Impact:** Reducing from 13k → 3.5k tokens saves \~73% in API costs and improves response latency by 60–70%.

---

## 📚 New Learnings & Best Practices

### 🔄 Handling Recursion Limits

* When deep agents or tools recurse too much, Python may throw a `RecursionError`.
* **Avoid simply increasing the recursion limit** — it may hide design flaws or cause infinite loops.
* Instead:

  * Set clear stopping conditions in agents.
  * Limit maximum depth of reasoning/tool calls.
  * Use guardrails in prompts to discourage looping.

### 📝 Prompt & Tool Definition Practices

* Keep prompts **precise** and **minimal**.
* Avoid duplicate phrases across instructions and sub-agent definitions (they bloat token count).
* Provide **short, structured tool descriptions** instead of verbose text.
* Define workflows clearly so the model doesn’t need extra clarification cycles.

#### Example 1: Context Engineering for Tool Descriptions

**Before (v1, verbose):**

```python
@tool
def get_stock_price(symbol: str) -> str:
    """This tool is designed to fetch stock-related data, including price, company name, market capitalization, P/E ratio, and 52-week highs and lows. Use this whenever you need to know about a company's stock."""
```

**After (v2, concise):**

```python
@tool
def get_stock_price(symbol: str) -> str:
    """Fetch stock price, company name, market cap, P/E ratio, 52-week range."""
```

➡️ The shorter description conveys the same meaning but reduces tokens and avoids repetition.

**Token Comparison:**

* v1: \~60 tokens
* v2: \~20 tokens
* **Reduction:** \~67%

#### Example 2: Context Engineering for Sub-Agent Prompts

**Before (v1, verbose):**

```python
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
```

**After (v2, concise):**

```python
fundamental_analyst = {
    "name": "fundamental-analyst",
    "description": "Performs company fundamental analysis",
    "prompt": "You are a fundamental analyst. Focus only on:\n- Financial statements (Revenue, Net Income, Assets, Debt)\n- Ratios: P/E, P/B, ROE, ROA, Debt/Equity\n- Growth trends vs peers\n- Valuation (intrinsic value)"
}
```

➡️ Context was streamlined by removing persona fluff ("expert with 15+ years") and redundant phrasing. The new version conveys the same scope but with fewer tokens.

**Token Comparison:**

* v1: \~120 tokens
* v2: \~55 tokens
* **Reduction:** \~55%

### Why Tool Descriptions Matter

* Tool descriptions are always loaded into the agent’s context.
* Long, verbose descriptions add up quickly because they are **repeated on every query**.
* Short, clear descriptions save tokens while still giving the model enough guidance.
* Example: If you have 5 tools and each has a 100-token description, that’s **500 tokens added to every query** — even if only 1 tool is used.

### ⚡ Controlling Context for Speed

* Externalize large prompts and configs into `.md` and `.json` files.
* Pass only the **needed tools** and **subagents** for each run.
* Remove unnecessary background text or repetition.
* Result: faster responses, fewer tokens, and no information loss.

---

## ✅ Quick Checklist

* [ ] Keep tool descriptions under 1–2 lines.
* [ ] Avoid persona fluff in sub-agent prompts.
* [ ] Externalize long instructions to `.md` and `.json` files.
* [ ] Monitor token counts with `tiktoken` or similar utilities.
* [ ] Set guardrails to prevent recursion loops.

---

## 🔮 Next Steps

* Add multi-company comparative analysis.
* Add streaming responses for real-time progress display.
* Extend UI with charts (price trends, RSI graphs).
* Option to export reports as PDF.

---

## ✅ Summary

Version 2 of the Stock Research Agent delivers:

* Cleaner configuration (`.env`, `.json`, `.md`).
* Flexible and optional search provider setup.
* Optimized token usage with **context engineering**.
* Practical learnings on recursion, prompt design, and context control.
* Much faster and cheaper queries without sacrificing depth.

This sets the foundation for more advanced reporting, visualization, and deployment.

---

## 📸 Screenshots & 🎥 Demo Video

* [Screenshot 1](./deep-research-agents-v2/screenshots/image1.png) – UI previews and analysis examples
* [Screenshot 2](./deep-research-agents-v2/screenshots/image2.png) – Example stock research report
* [Demo Video](./deep-research-agents-v2/deepagents-v2.mp4) – walkthrough of the stock research workflow

---

## 📖 Reference

* [Version 1 README](/deep-research-agents-v1/readme.md) – for comparison with the original implementation


## 🚨 Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.

---

## 🙏 Acknowledgments

* **LangChain** – for the DeepAgent framework
* **Yahoo Finance** – for free and reliable financial data APIs
* **Gradio** – for the intuitive and flexible UI framework
* **Ollama** – for enabling local LLM hosting
* **Tavily** – for providing fast browser search APIs
* **Brave** – for offering a robust web search API

---
## 🌟 Star the Project

If you find this project useful, please consider giving it a star ⭐️ on GitHub!

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sagar-n/deepagents&type=Date)](https://star-history.com/#sagar-n/deepagents&Date)

**Built with ❤️ using LangChain DeepAgents**

*Transform your investment research with the power of specialized AI agents.*


