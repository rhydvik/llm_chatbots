"""
Chat router for the chatbot boilerplate.
Handles chat interactions with session management.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.agents.chat_agent import ChatAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="Session ID (will create new if not provided)")
    user_type: Optional[str] = Field("customer", description="User type for specialized responses")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI's response")
    session_id: str = Field(..., description="Session identifier")
    user_type: str = Field(..., description="User type used for this chat")
    is_new_session: bool = Field(..., description="Whether this was a new session")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

chat_agent = None

async def get_chat_agent() -> ChatAgent:
    """Get the chat agent instance."""
    global chat_agent
    if chat_agent is None:
        chat_agent = ChatAgent()
        await chat_agent.initialize()
    return chat_agent

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: ChatAgent = Depends(get_chat_agent)
):
    """
    Main chat endpoint.
    
    Handles both new and existing chat sessions:
    - If session_id is provided and exists: continues the conversation
    - If session_id is not provided or doesn't exist: starts a new session
    """
    try:
        if not request.session_id:
            session_id = str(uuid.uuid4())
            is_new_session = True
        else:
            session_id = request.session_id
            is_new_session = not await agent.session_exists(session_id)
        
        logger.info(f"Chat request - Session: {session_id}, New: {is_new_session}, Message: {request.message[:50]}...")
        
        response = await agent.chat(
            message=request.message,
            session_id=session_id,
            user_type=request.user_type
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            user_type=request.user_type,
            is_new_session=is_new_session,
            metadata={
                "message_length": len(request.message),
                "response_length": len(response),
                "processing_time": "calculated_later"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/sessions/{session_id}/history")
async def get_chat_history(
    session_id: str,
    agent: ChatAgent = Depends(get_chat_agent)
):
    """Get conversation history for a session."""
    try:
        history = await agent.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    agent: ChatAgent = Depends(get_chat_agent)
):
    """Clear a chat session."""
    try:
        await agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")

@router.get("/health")
async def chat_health():
    """Health check for chat service."""
    from app.utils.llm_provider import get_available_providers, validate_provider_config
    import os
    
    current_provider = os.getenv("LLM_PROVIDER", "openai")
    available_providers = get_available_providers()
    provider_status = validate_provider_config(current_provider)
    
    return {
        "status": "healthy",
        "service": "chat",
        "agent_initialized": chat_agent is not None,
        "llm_config": {
            "current_provider": current_provider,
            "available_providers": available_providers,
            "provider_configured": provider_status["configured"],
            "missing_config": provider_status.get("missing", [])
        },
        "endpoints": {
            "chat": "/chat/",
            "history": "/chat/sessions/{session_id}/history",
            "clear": "/chat/sessions/{session_id}"
        }
    } 