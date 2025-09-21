import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model selection
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")

    # Ollama
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20B")

    # LM Studio
    LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")
    LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

    # Search APIs
    BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Gradio
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")


    # Fundamental Analyst
    FUNDAMENTAL_ANALYST = {
        "name": os.getenv("FUNDAMENTAL_ANALYST_NAME", "fundamental-analyst"),
        "description": os.getenv("FUNDAMENTAL_ANALYST_DESC", ""),
        "prompt": os.getenv("FUNDAMENTAL_ANALYST_PROMPT", "")
    }

    # Technical Analyst
    TECHNICAL_ANALYST = {
        "name": os.getenv("TECHNICAL_ANALYST_NAME", "technical-analyst"),
        "description": os.getenv("TECHNICAL_ANALYST_DESC", ""),
        "prompt": os.getenv("TECHNICAL_ANALYST_PROMPT", "")
    }

    # Risk Analyst
    RISK_ANALYST = {
        "name": os.getenv("RISK_ANALYST_NAME", "risk-analyst"),
        "description": os.getenv("RISK_ANALYST_DESC", ""),
        "prompt": os.getenv("RISK_ANALYST_PROMPT", "")
    }

    SUBAGENTS = [FUNDAMENTAL_ANALYST, TECHNICAL_ANALYST, RISK_ANALYST]

    # Agent instructions
    # RESEARCH_INSTRUCTIONS = os.getenv("RESEARCH_INSTRUCTIONS", "")

    RESEARCH_INSTRUCTIONS = os.getenv("RESEARCH_INSTRUCTIONS", "").replace("\\n", "\n") 