"""
Simple chat agent using LangGraph with OpenAI integration.
This is a minimal implementation to get started.
"""

import logging
import os
from typing import Dict, List, Optional, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from .states.chat_state import SessionInfo
from .nodes.chat_nodes import ChatNodes
from .graph.chat_graph import ChatGraphBuilder
from app.utils.llm_provider import get_llm

logger = logging.getLogger(__name__)


class ChatAgent:
    """
    A simple chat agent using LangGraph with OpenAI integration.
    
    Graph flow:
    1. Input Processing -> 2. LLM Processing -> 3. Response Formatting
    """
    
    def __init__(self):
        self.graph = None
        self.llm = None
        self.memory = MemorySaver()
        self.sessions: Dict[str, SessionInfo] = {}
        self.chat_nodes = None
        self.graph_builder = None
        
    async def initialize(self):
        """Initialize the chat agent."""
        logger.info("Initializing chat agent...")
        
        self.llm = get_llm()
        
        if self.llm is None:
            logger.warning("No LLM configured. Running in mock mode.")
        
        self.chat_nodes = ChatNodes(llm=self.llm, sessions=self.sessions)
        self.graph_builder = ChatGraphBuilder(self.chat_nodes)
        
        self.graph = self.graph_builder.build_graph(self.memory)
        
        logger.info("Chat agent initialized successfully")
    
    async def chat(self, message: str, session_id: str = "default", user_type: str = "customer") -> str:
        """
        Main chat interface.
        
        Args:
            message: User message
            session_id: Session identifier
            user_type: User type for specialized responses
            
        Returns:
            AI response
        """
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "user_type": user_type,
                "processed": False
            }
            
            thread_config = {"configurable": {"thread_id": session_id}}
            
            result = await self.graph.ainvoke(initial_state, thread_config)
            
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Chat processing error: {e}", exc_info=True)
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        try:
            thread_config = {"configurable": {"thread_id": session_id}}
            history = []
            
            current_state = await self.graph.aget_state(thread_config)
            if current_state and current_state.values.get("messages"):
                for msg in current_state.values["messages"]:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        history.append({
                            "type": "human" if isinstance(msg, HumanMessage) else "ai",
                            "content": msg.content,
                            "timestamp": getattr(msg, 'timestamp', None)
                        })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def clear_session(self, session_id: str):
        """Clear a conversation session."""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            logger.info(f"Session {session_id} cleared")
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
    
    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        return self.sessions.get(session_id) 