"""
LLM Provider management utilities.
Supports multiple LLM providers with dynamic loading.
"""

import logging
import os

# Core LangChain
from langchain_openai import ChatOpenAI

# Multi-provider imports (install dependencies as needed)
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
    
try:
    from langchain_groq import ChatGroq  
except ImportError:
    ChatGroq = None
    
try:
    from langchain_google_genai import ChatGoogleGenerativeAI as ChatGemini
except ImportError:
    ChatGemini = None
    
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

logger = logging.getLogger(__name__)


def get_llm():
    """Get the LLM based on environment configuration."""
    
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Using placeholder.")
            return None
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000, api_key=api_key)
    
    elif llm_provider == "anthropic":
        if ChatAnthropic is None:
            logger.error("langchain_anthropic not installed. Run: pip install langchain_anthropic")
            return None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found. Using placeholder.")
            return None
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=1000, api_key=api_key)
    
    elif llm_provider == "groq":
        if ChatGroq is None:
            logger.error("langchain_groq not installed. Run: pip install langchain_groq")
            return None
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found. Using placeholder.")
            return None
        return ChatGroq(model="llama3-8b-8192", temperature=0.7, max_tokens=1000, api_key=api_key)
    
    elif llm_provider == "gemini":
        if ChatGemini is None:
            logger.error("langchain_google_genai not installed. Run: pip install langchain_google_genai")
            return None
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Using placeholder.")
            return None
        return ChatGemini(model="gemini-2.0-flash-exp", temperature=0.7, max_tokens=1000, google_api_key=api_key)
    
    elif llm_provider == "ollama":
        if ChatOllama is None:
            logger.error("langchain_community not available for ChatOllama")
            return None
        # Ollama typically runs locally, no API key needed
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model="llama3.2", base_url=base_url, temperature=0.7)
    
    else:
        logger.warning(f"Unknown LLM provider '{llm_provider}'. Defaulting to OpenAI.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Using placeholder.")
            return None
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000, api_key=api_key)


def get_available_providers():
    """Get list of available LLM providers based on installed packages."""
    providers = ["openai"]  # Always available
    
    if ChatAnthropic is not None:
        providers.append("anthropic")
    if ChatGroq is not None:
        providers.append("groq")
    if ChatGemini is not None:
        providers.append("gemini")
    if ChatOllama is not None:
        providers.append("ollama")
        
    return providers


def validate_provider_config(provider: str) -> dict:
    """Validate that required configuration exists for a provider."""
    provider = provider.lower()
    status = {"provider": provider, "configured": False, "missing": []}
    
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            status["missing"].append("OPENAI_API_KEY")
        else:
            status["configured"] = True
            
    elif provider == "anthropic":
        if ChatAnthropic is None:
            status["missing"].append("langchain_anthropic package")
        if not os.getenv("ANTHROPIC_API_KEY"):
            status["missing"].append("ANTHROPIC_API_KEY")
        if not status["missing"]:
            status["configured"] = True
            
    elif provider == "groq":
        if ChatGroq is None:
            status["missing"].append("langchain_groq package")
        if not os.getenv("GROQ_API_KEY"):
            status["missing"].append("GROQ_API_KEY")
        if not status["missing"]:
            status["configured"] = True
            
    elif provider == "gemini":
        if ChatGemini is None:
            status["missing"].append("langchain_google_genai package")
        if not os.getenv("GOOGLE_API_KEY"):
            status["missing"].append("GOOGLE_API_KEY")
        if not status["missing"]:
            status["configured"] = True
            
    elif provider == "ollama":
        if ChatOllama is None:
            status["missing"].append("langchain_community package")
        # Ollama doesn't require API key, just check if service is available
        if not status["missing"]:
            status["configured"] = True
    
    return status