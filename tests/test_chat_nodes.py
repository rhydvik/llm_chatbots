"""
Tests for ChatNodes processing functionality.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.nodes.chat_nodes import ChatNodes
from app.agents.states.chat_state import SessionInfo


class TestChatNodes:
    """Test cases for ChatNodes class."""
    
    def test_chat_nodes_initialization_with_llm(self, mock_openai_llm, mock_sessions):
        """Test ChatNodes initialization with LLM."""
        nodes = ChatNodes(llm=mock_openai_llm, sessions=mock_sessions)
        assert nodes.llm == mock_openai_llm
        assert nodes.sessions == mock_sessions
        
    def test_chat_nodes_initialization_without_llm(self, mock_sessions):
        """Test ChatNodes initialization without LLM."""
        nodes = ChatNodes(llm=None, sessions=mock_sessions)
        assert nodes.llm is None
        assert nodes.sessions == mock_sessions
        
    def test_chat_nodes_initialization_defaults(self):
        """Test ChatNodes initialization with defaults."""
        nodes = ChatNodes()
        assert nodes.llm is None
        assert nodes.sessions == {}


class TestInputProcessingNode:
    """Test cases for input processing node."""
    
    @pytest.mark.asyncio
    async def test_input_processing_adds_system_message(self, chat_nodes, sample_chat_state):
        """Test that input processing adds system message for customer."""
        sample_chat_state["user_type"] = "customer"
        result = await chat_nodes.input_processing_node(sample_chat_state)
        
        # Check that system message was added
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], SystemMessage)
        assert "helpful AI assistant for customers" in result["messages"][0].content
        
    @pytest.mark.asyncio
    async def test_input_processing_support_agent_prompt(self, chat_nodes, sample_chat_state):
        """Test input processing with support agent user type."""
        sample_chat_state["user_type"] = "support_agent"
        result = await chat_nodes.input_processing_node(sample_chat_state)
        
        assert isinstance(result["messages"][0], SystemMessage)
        assert "AI assistant for support agents" in result["messages"][0].content
        
    @pytest.mark.asyncio
    async def test_input_processing_manager_prompt(self, chat_nodes, sample_chat_state):
        """Test input processing with manager user type."""
        sample_chat_state["user_type"] = "manager"
        result = await chat_nodes.input_processing_node(sample_chat_state)
        
        assert isinstance(result["messages"][0], SystemMessage)
        assert "AI assistant for managers" in result["messages"][0].content
        
    @pytest.mark.asyncio
    async def test_input_processing_preserves_existing_system_message(self, chat_nodes):
        """Test that existing system message is preserved."""
        state = {
            "messages": [
                SystemMessage(content="Custom system message"),
                HumanMessage(content="Hello")
            ],
            "session_id": "test",
            "user_type": "customer",
            "processed": False
        }
        
        result = await chat_nodes.input_processing_node(state)
        
        # Should not add another system message
        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "Custom system message"
        
    @pytest.mark.asyncio
    async def test_input_processing_unknown_user_type(self, chat_nodes, sample_chat_state):
        """Test input processing with unknown user type defaults to customer."""
        sample_chat_state["user_type"] = "unknown_type"
        result = await chat_nodes.input_processing_node(sample_chat_state)
        
        assert isinstance(result["messages"][0], SystemMessage)
        assert "helpful AI assistant for customers" in result["messages"][0].content


class TestLLMProcessingNode:
    """Test cases for LLM processing node."""
    
    @pytest.mark.asyncio
    async def test_llm_processing_with_mock_llm(self, chat_nodes, sample_chat_state):
        """Test LLM processing with mocked LLM."""
        result = await chat_nodes.llm_processing_node(sample_chat_state)
        
        # Should have original message plus AI response
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "Mock AI response"
        
    @pytest.mark.asyncio
    async def test_llm_processing_without_llm(self, chat_nodes_no_llm, sample_chat_state):
        """Test LLM processing without LLM (fallback behavior)."""
        result = await chat_nodes_no_llm.llm_processing_node(sample_chat_state)
        
        # Should have original message plus fallback response
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert "mock chatbot response" in result["messages"][-1].content
        assert "without OpenAI API key" in result["messages"][-1].content
        
    @pytest.mark.asyncio
    async def test_llm_processing_includes_user_type_in_fallback(self, chat_nodes_no_llm, sample_chat_state):
        """Test that fallback response includes user type."""
        sample_chat_state["user_type"] = "manager"
        result = await chat_nodes_no_llm.llm_processing_node(sample_chat_state)
        
        assert "User type: manager" in result["messages"][-1].content
        
    @pytest.mark.asyncio
    async def test_llm_processing_error_handling(self, mock_sessions):
        """Test LLM processing error handling."""
        # Create a mock LLM that raises an exception
        error_llm = Mock(spec=ChatOpenAI)
        error_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        nodes = ChatNodes(llm=error_llm, sessions=mock_sessions)
        
        state = {
            "messages": [HumanMessage(content="Test")],
            "session_id": "error_test",
            "user_type": "customer",
            "processed": False
        }
        
        result = await nodes.llm_processing_node(state)
        
        # Should handle error gracefully
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][-1], AIMessage)
        assert "technical difficulties" in result["messages"][-1].content


class TestResponseFormattingNode:
    """Test cases for response formatting node."""
    
    @pytest.mark.asyncio
    async def test_response_formatting_marks_processed(self, chat_nodes, sample_chat_state):
        """Test that response formatting marks state as processed."""
        assert sample_chat_state["processed"] is False
        
        result = await chat_nodes.response_formatting_node(sample_chat_state)
        
        assert result["processed"] is True
        
    @pytest.mark.asyncio
    async def test_response_formatting_creates_new_session(self, chat_nodes, sample_chat_state):
        """Test response formatting creates new session info."""
        session_id = sample_chat_state["session_id"]
        assert session_id not in chat_nodes.sessions
        
        await chat_nodes.response_formatting_node(sample_chat_state)
        
        assert session_id in chat_nodes.sessions
        assert chat_nodes.sessions[session_id].message_count == 1
        assert chat_nodes.sessions[session_id].user_type == sample_chat_state["user_type"]
        
    @pytest.mark.asyncio
    async def test_response_formatting_updates_existing_session(self, chat_nodes, sample_chat_state):
        """Test response formatting updates existing session."""
        session_id = sample_chat_state["session_id"]
        
        # Pre-populate session
        chat_nodes.sessions[session_id] = SessionInfo(
            session_id=session_id,
            message_count=5,
            user_type="customer"
        )
        
        await chat_nodes.response_formatting_node(sample_chat_state)
        
        # Should increment message count
        assert chat_nodes.sessions[session_id].message_count == 6
        
    @pytest.mark.asyncio
    async def test_response_formatting_different_user_types(self, chat_nodes):
        """Test response formatting with different user types."""
        user_types = ["customer", "support_agent", "manager"]
        
        for i, user_type in enumerate(user_types):
            state = {
                "messages": [HumanMessage(content="Test")],
                "session_id": f"test_session_{i}",
                "user_type": user_type,
                "processed": False
            }
            
            await chat_nodes.response_formatting_node(state)
            
            session_id = state["session_id"]
            assert chat_nodes.sessions[session_id].user_type == user_type


class TestChatNodesIntegration:
    """Integration tests for ChatNodes workflow."""
    
    @pytest.mark.asyncio
    async def test_full_node_workflow(self, chat_nodes):
        """Test complete workflow through all nodes."""
        initial_state = {
            "messages": [HumanMessage(content="Hello, I need help!")],
            "session_id": "integration_test",
            "user_type": "customer",
            "processed": False
        }
        
        # Process through all nodes
        state_after_input = await chat_nodes.input_processing_node(initial_state)
        state_after_llm = await chat_nodes.llm_processing_node(state_after_input)
        final_state = await chat_nodes.response_formatting_node(state_after_llm)
        
        # Verify final state
        assert final_state["processed"] is True
        assert len(final_state["messages"]) == 3  # System, Human, AI
        assert isinstance(final_state["messages"][0], SystemMessage)
        assert isinstance(final_state["messages"][1], HumanMessage)
        assert isinstance(final_state["messages"][2], AIMessage)
        
        # Verify session was created
        session_id = final_state["session_id"]
        assert session_id in chat_nodes.sessions
        assert chat_nodes.sessions[session_id].message_count == 1 