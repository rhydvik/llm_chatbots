"""
Chat agent state definitions.
"""

from typing import Sequence, TypedDict, Annotated
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """State for the simple chat agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    user_type: str
    processed: bool


@dataclass
class SessionInfo:
    """Simple session information tracking."""
    session_id: str
    message_count: int = 0
    user_type: str = "customer" 