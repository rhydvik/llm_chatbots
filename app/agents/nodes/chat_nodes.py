"""
Chat agent node functions for LangGraph processing.
"""

import logging
from typing import Dict

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from ..states.chat_state import ChatState, SessionInfo
from ..utils.chat_utils import get_system_prompt

logger = logging.getLogger(__name__)


class ChatNodes:
    """Container class for chat processing nodes."""
    
    def __init__(self, llm: ChatOpenAI = None, sessions: Dict[str, SessionInfo] = None):
        self.llm = llm
        self.sessions = sessions if sessions is not None else {}
    
    async def input_processing_node(self, state: ChatState) -> ChatState:
        """Node for processing user input and preparing context."""
        logger.info(f"Processing input for session: {state['session_id']}")
        
        system_prompt = get_system_prompt(state["user_type"])
        
        has_system_message = any(isinstance(msg, SystemMessage) for msg in state["messages"])
        
        if not has_system_message:
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
            state["messages"] = messages
        
        return state
    
    async def llm_processing_node(self, state: ChatState) -> ChatState:
        """Node for LLM processing - this connects to OpenAI."""
        logger.info(f"LLM processing for session: {state['session_id']}")
        
        try:
            if self.llm is None:
                response_content = f"Hello! I'm a mock chatbot response to your message. (User type: {state['user_type']}) Your message was processed successfully, but I'm running without OpenAI API key. Please configure OPENAI_API_KEY environment variable for real AI responses."
                response = AIMessage(content=response_content)
            else:
                response = await self.llm.ainvoke(state["messages"])
            
            messages = list(state["messages"])
            messages.append(response)
            state["messages"] = messages
            
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            error_message = AIMessage(
                content="I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
            )
            messages = list(state["messages"])
            messages.append(error_message)
            state["messages"] = messages
        
        return state
    
    async def response_formatting_node(self, state: ChatState) -> ChatState:
        """Node for formatting the final response."""
        logger.info(f"Formatting response for session: {state['session_id']}")
        
        state["processed"] = True
        
        session_id = state["session_id"]
        if session_id in self.sessions:
            self.sessions[session_id].message_count += 1
        else:
            self.sessions[session_id] = SessionInfo(
                session_id=session_id,
                message_count=1,
                user_type=state["user_type"]
            )
        
        return state 