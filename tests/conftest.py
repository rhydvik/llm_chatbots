"""
Test configuration and fixtures for chatbot boilerplate tests.
"""

import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agents.chat_agent import ChatAgent
from app.agents.states.chat_state import SessionInfo, ChatState
from app.agents.nodes.chat_nodes import ChatNodes
from app.agents.graph.chat_graph import ChatGraphBuilder


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    mock_llm = Mock(spec=ChatOpenAI)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock AI response"))
    return mock_llm


@pytest.fixture
def mock_sessions():
    """Mock sessions dictionary for testing."""
    return {}


@pytest.fixture
def chat_nodes(mock_openai_llm, mock_sessions):
    """Create ChatNodes instance for testing."""
    return ChatNodes(llm=mock_openai_llm, sessions=mock_sessions)


@pytest.fixture
def chat_nodes_no_llm(mock_sessions):
    """Create ChatNodes instance without LLM for testing fallback behavior."""
    return ChatNodes(llm=None, sessions=mock_sessions)


@pytest.fixture
def graph_builder(chat_nodes):
    """Create ChatGraphBuilder instance for testing."""
    return ChatGraphBuilder(chat_nodes)


@pytest.fixture
def memory_saver():
    """Create MemorySaver instance for testing."""
    return MemorySaver()


@pytest.fixture
def sample_chat_state():
    """Create sample ChatState for testing."""
    return {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "session_id": "test_session_123",
        "user_type": "customer",
        "processed": False
    }


@pytest.fixture
def sample_session_info():
    """Create sample SessionInfo for testing."""
    return SessionInfo(
        session_id="test_session_123",
        message_count=5,
        user_type="customer"
    )


@pytest_asyncio.fixture
async def initialized_chat_agent():
    """Create and initialize a ChatAgent for testing."""
    # Temporarily set mock API key for testing
    original_api_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    
    # Mock the ChatOpenAI class to avoid actual API calls
    with patch("app.agents.chat_agent.ChatOpenAI") as mock_openai_class:
        mock_llm = Mock(spec=ChatOpenAI)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        mock_openai_class.return_value = mock_llm
        
        agent = ChatAgent()
        await agent.initialize()
        
        # Verify the agent is properly mocked
        assert agent.llm is not None
        assert agent.chat_nodes is not None
        assert agent.graph is not None
    
    yield agent
    
    # Restore original API key
    if original_api_key:
        os.environ["OPENAI_API_KEY"] = original_api_key
    elif "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]


@pytest_asyncio.fixture
async def chat_agent_no_api_key():
    """Create ChatAgent without API key for testing fallback behavior."""
    # Ensure no API key is set
    original_api_key = os.environ.get("OPENAI_API_KEY")
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    
    agent = ChatAgent()
    await agent.initialize()
    
    yield agent
    
    # Restore original API key
    if original_api_key:
        os.environ["OPENAI_API_KEY"] = original_api_key


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I help you today?"),
        HumanMessage(content="What's the weather like?")
    ]


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables after each test."""
    yield
    # Clean up any test-specific environment variables if needed 