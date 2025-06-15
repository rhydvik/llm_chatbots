"""
Chat agent utility functions.
"""

import logging

logger = logging.getLogger(__name__)


def get_system_prompt(user_type: str) -> str:
    """Get system prompt based on user type."""
    prompts = {
        "customer": "You are a helpful AI assistant for customers. Be friendly, clear, and helpful in solving their needs.",
        "support_agent": "You are an AI assistant for support agents. Provide detailed, accurate information to help resolve customer issues.",
        "manager": "You are an AI assistant for managers. Provide strategic insights and data-driven recommendations."
    }
    return prompts.get(user_type, prompts["customer"]) 