## ▶️ How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/stock-research-agent.git
cd stock-research-agent
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up your `.env` file:**

```env
TAVILY_API_KEY=your_tavily_key_here
BRAVE_SEARCH_API_KEY=your_brave_key_here
OLLAMA_MODEL=gpt-oss:20B
LM_STUDIO_MODEL=local-model
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
DEFAULT_MODEL_PROVIDER=ollama
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=8001
RECURSION_LIMIT = 30
```

5. **Run the app:**

```bash
python research-agents-v2.py
```

The Gradio interface will launch at `http://0.0.0.0:8001/`

---
