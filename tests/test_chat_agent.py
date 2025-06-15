"""
Tests for ChatAgent class - the main orchestrator.
"""

import os
import pytest
from unittest.mock import AsyncMock, Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agents.chat_agent import ChatAgent
from app.agents.states.chat_state import SessionInfo


class TestChatAgentInitialization:
    """Test cases for ChatAgent initialization."""
    
    def test_chat_agent_creation(self):
        """Test basic ChatAgent creation."""
        agent = ChatAgent()
        assert agent.graph is None
        assert agent.llm is None
        assert agent.memory is not None
        assert agent.sessions == {}
        assert agent.chat_nodes is None
        assert agent.graph_builder is None
        
    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self):
        """Test initialization with API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            agent = ChatAgent()
            
            # Mock the ChatOpenAI to avoid actual API calls
            with patch("app.agents.chat_agent.ChatOpenAI") as mock_openai:
                mock_llm = Mock()
                mock_openai.return_value = mock_llm
                
                await agent.initialize()
                
                assert agent.llm == mock_llm
                assert agent.chat_nodes is not None
                assert agent.graph_builder is not None
                assert agent.graph is not None
                
    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization without API key."""
        # Ensure no API key is set
        with patch.dict(os.environ, {}, clear=True):
            agent = ChatAgent()
            await agent.initialize()
            
            assert agent.llm is None
            assert agent.chat_nodes is not None
            assert agent.graph_builder is not None
            assert agent.graph is not None
            
    @pytest.mark.asyncio
    async def test_initialize_creates_graph_components(self):
        """Test that initialization creates all necessary components."""
        agent = ChatAgent()
        await agent.initialize()
        
        # All components should be created
        assert agent.chat_nodes is not None
        assert agent.graph_builder is not None
        assert agent.graph is not None
        assert agent.memory is not None


class TestChatFunctionality:
    """Test cases for chat functionality."""
    
    @pytest.mark.asyncio
    async def test_chat_basic_message(self, initialized_chat_agent):
        """Test basic chat functionality."""
        response = await initialized_chat_agent.chat(
            message="Hello, how are you?",
            session_id="test_session",
            user_type="customer"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    @pytest.mark.asyncio
    async def test_chat_different_user_types(self, initialized_chat_agent):
        """Test chat with different user types."""
        user_types = ["customer", "support_agent", "manager"]
        
        for user_type in user_types:
            response = await initialized_chat_agent.chat(
                message=f"Hello as {user_type}",
                session_id=f"session_{user_type}",
                user_type=user_type
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            
    @pytest.mark.asyncio
    async def test_chat_default_parameters(self, initialized_chat_agent):
        """Test chat with default parameters."""
        response = await initialized_chat_agent.chat("Hello!")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    @pytest.mark.asyncio
    async def test_chat_creates_session(self, initialized_chat_agent):
        """Test that chat creates session entries."""
        session_id = "new_session_test"
        
        print(f"\nBefore chat - Sessions: {initialized_chat_agent.sessions}")
        print(f"ChatNodes sessions: {initialized_chat_agent.chat_nodes.sessions}")
        print(f"Same object? {initialized_chat_agent.sessions is initialized_chat_agent.chat_nodes.sessions}")
        
        response = await initialized_chat_agent.chat(
            message="Hello",
            session_id=session_id,
            user_type="customer"
        )
        
        print(f"Response: {response}")
        print(f"After chat - Sessions: {initialized_chat_agent.sessions}")
        print(f"ChatNodes sessions: {initialized_chat_agent.chat_nodes.sessions}")
        
        assert session_id in initialized_chat_agent.sessions
        session_info = initialized_chat_agent.sessions[session_id]
        assert session_info.session_id == session_id
        assert session_info.user_type == "customer"
        
    @pytest.mark.asyncio
    async def test_chat_error_handling(self):
        """Test chat error handling."""
        agent = ChatAgent()
        
        # Mock a failing graph
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(side_effect=Exception("Graph error"))
        agent.graph = mock_graph
        
        response = await agent.chat("Hello")
        
        # Should return error message gracefully
        assert "technical difficulties" in response.lower()
        
    @pytest.mark.asyncio
    async def test_chat_no_ai_response(self):
        """Test chat when no AI response is generated."""
        agent = ChatAgent()
        
        # Mock a graph that returns no AI messages
        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [HumanMessage(content="Hello")]
        })
        agent.graph = mock_graph
        
        response = await agent.chat("Hello")
        
        # Should return fallback message
        assert "couldn't generate a response" in response


class TestSessionManagement:
    """Test cases for session management."""
    
    @pytest.mark.asyncio
    async def test_session_exists(self, initialized_chat_agent):
        """Test session existence checking."""
        session_id = "existence_test"
        
        # Initially should not exist
        exists = await initialized_chat_agent.session_exists(session_id)
        assert not exists
        
        # After chat, should exist
        await initialized_chat_agent.chat("Hello", session_id=session_id)
        exists = await initialized_chat_agent.session_exists(session_id)
        assert exists
        
    @pytest.mark.asyncio
    async def test_get_session_info(self, initialized_chat_agent):
        """Test getting session information."""
        session_id = "info_test"
        
        # Initially should return None
        info = await initialized_chat_agent.get_session_info(session_id)
        assert info is None
        
        # After chat, should return session info
        await initialized_chat_agent.chat("Hello", session_id=session_id, user_type="manager")
        info = await initialized_chat_agent.get_session_info(session_id)
        
        assert info is not None
        assert info.session_id == session_id
        assert info.user_type == "manager"
        assert info.message_count >= 1
        
    @pytest.mark.asyncio
    async def test_clear_session(self, initialized_chat_agent):
        """Test clearing sessions."""
        session_id = "clear_test"
        
        # Create session
        await initialized_chat_agent.chat("Hello", session_id=session_id)
        assert await initialized_chat_agent.session_exists(session_id)
        
        # Clear session
        await initialized_chat_agent.clear_session(session_id)
        
        # Should no longer exist
        assert not await initialized_chat_agent.session_exists(session_id)
        
    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, initialized_chat_agent):
        """Test clearing a session that doesn't exist."""
        # Should not raise an error
        await initialized_chat_agent.clear_session("nonexistent_session")
        
    @pytest.mark.asyncio
    async def test_multiple_sessions(self, initialized_chat_agent):
        """Test managing multiple sessions."""
        sessions = ["session1", "session2", "session3"]
        
        # Create multiple sessions
        for session_id in sessions:
            await initialized_chat_agent.chat(f"Hello {session_id}", session_id=session_id)
            
        # All should exist
        for session_id in sessions:
            assert await initialized_chat_agent.session_exists(session_id)
            
        # Clear one session
        await initialized_chat_agent.clear_session(sessions[1])
        
        # Check states
        assert await initialized_chat_agent.session_exists(sessions[0])
        assert not await initialized_chat_agent.session_exists(sessions[1])
        assert await initialized_chat_agent.session_exists(sessions[2])


class TestConversationHistory:
    """Test cases for conversation history."""
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, initialized_chat_agent):
        """Test getting history for empty session."""
        history = await initialized_chat_agent.get_conversation_history("empty_session")
        assert history == []
        
    @pytest.mark.asyncio
    async def test_get_conversation_history_with_messages(self, initialized_chat_agent):
        """Test getting conversation history with messages."""
        session_id = "history_test"
        
        # Have a conversation
        await initialized_chat_agent.chat("Hello!", session_id=session_id)
        await initialized_chat_agent.chat("How are you?", session_id=session_id)
        
        history = await initialized_chat_agent.get_conversation_history(session_id)
        
        # Should have messages
        assert len(history) > 0
        
        # Check message structure
        for msg in history:
            assert "type" in msg
            assert "content" in msg
            assert msg["type"] in ["human", "ai"]
            
    @pytest.mark.asyncio
    async def test_get_conversation_history_error_handling(self):
        """Test conversation history error handling."""
        agent = ChatAgent()
        
        # Mock a failing graph
        mock_graph = Mock()
        mock_graph.aget_state = AsyncMock(side_effect=Exception("State error"))
        agent.graph = mock_graph
        
        history = await agent.get_conversation_history("error_session")
        
        # Should return empty list on error
        assert history == []


class TestChatAgentIntegration:
    """Integration tests for complete ChatAgent functionality."""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, initialized_chat_agent):
        """Test a complete conversation flow."""
        session_id = "full_conversation"
        
        # Start conversation
        response1 = await initialized_chat_agent.chat(
            "Hello, I need help with my account",
            session_id=session_id,
            user_type="customer"
        )
        assert isinstance(response1, str)
        
        # Continue conversation
        response2 = await initialized_chat_agent.chat(
            "Can you tell me my account balance?",
            session_id=session_id,
            user_type="customer"
        )
        assert isinstance(response2, str)
        
        # Check session was maintained
        assert await initialized_chat_agent.session_exists(session_id)
        session_info = await initialized_chat_agent.get_session_info(session_id)
        assert session_info.message_count >= 2
        
        # Get conversation history
        history = await initialized_chat_agent.get_conversation_history(session_id)
        assert len(history) >= 2
        
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, initialized_chat_agent):
        """Test handling concurrent sessions."""
        import asyncio
        
        async def chat_session(session_id, user_type):
            return await initialized_chat_agent.chat(
                f"Hello from {session_id}",
                session_id=session_id,
                user_type=user_type
            )
        
        # Create multiple concurrent sessions
        tasks = [
            chat_session("concurrent1", "customer"),
            chat_session("concurrent2", "support_agent"),
            chat_session("concurrent3", "manager")
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
            
        # All sessions should exist
        for i in range(1, 4):
            assert await initialized_chat_agent.session_exists(f"concurrent{i}")
            
    @pytest.mark.asyncio
    async def test_agent_without_openai_key(self, chat_agent_no_api_key):
        """Test agent functionality without OpenAI API key."""
        response = await chat_agent_no_api_key.chat(
            "Hello, test without API key",
            session_id="no_api_key_test",
            user_type="customer"
        )
        
        # Should still work with fallback responses
        assert isinstance(response, str)
        assert len(response) > 0
        assert "mock" in response.lower() or "api key" in response.lower() 