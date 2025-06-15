"""
Tests for chat state classes and functionality.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agents.states.chat_state import ChatState, SessionInfo


class TestSessionInfo:
    """Test cases for SessionInfo dataclass."""
    
    def test_session_info_creation(self):
        """Test basic SessionInfo creation."""
        session = SessionInfo(session_id="test_123")
        assert session.session_id == "test_123"
        assert session.message_count == 0
        assert session.user_type == "customer"
        
    def test_session_info_with_values(self):
        """Test SessionInfo creation with custom values."""
        session = SessionInfo(
            session_id="test_456",
            message_count=5,
            user_type="manager"
        )
        assert session.session_id == "test_456"
        assert session.message_count == 5
        assert session.user_type == "manager"
        
    def test_session_info_equality(self):
        """Test SessionInfo equality comparison."""
        session1 = SessionInfo(session_id="test", message_count=3, user_type="customer")
        session2 = SessionInfo(session_id="test", message_count=3, user_type="customer")
        session3 = SessionInfo(session_id="different", message_count=3, user_type="customer")
        
        assert session1 == session2
        assert session1 != session3
        
    def test_session_info_modification(self):
        """Test SessionInfo field modification."""
        session = SessionInfo(session_id="test")
        assert session.message_count == 0
        
        session.message_count = 10
        assert session.message_count == 10
        
        session.user_type = "support_agent"
        assert session.user_type == "support_agent"


class TestChatState:
    """Test cases for ChatState TypedDict functionality."""
    
    def test_chat_state_creation(self):
        """Test basic ChatState creation."""
        messages = [HumanMessage(content="Hello")]
        state: ChatState = {
            "messages": messages,
            "session_id": "test_session",
            "user_type": "customer",
            "processed": False
        }
        
        assert state["messages"] == messages
        assert state["session_id"] == "test_session"
        assert state["user_type"] == "customer"
        assert state["processed"] is False
        
    def test_chat_state_with_multiple_messages(self):
        """Test ChatState with multiple messages."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!")
        ]
        
        state: ChatState = {
            "messages": messages,
            "session_id": "multi_msg_session",
            "user_type": "support_agent",
            "processed": False
        }
        
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], SystemMessage)
        assert isinstance(state["messages"][1], HumanMessage)
        assert isinstance(state["messages"][2], AIMessage)
        
    def test_chat_state_message_content(self):
        """Test ChatState message content access."""
        messages = [
            HumanMessage(content="Test message"),
            AIMessage(content="Test response")
        ]
        
        state: ChatState = {
            "messages": messages,
            "session_id": "content_test",
            "user_type": "customer",
            "processed": True
        }
        
        assert state["messages"][0].content == "Test message"
        assert state["messages"][1].content == "Test response"
        assert state["processed"] is True
        
    def test_chat_state_user_types(self):
        """Test ChatState with different user types."""
        user_types = ["customer", "support_agent", "manager"]
        
        for user_type in user_types:
            state: ChatState = {
                "messages": [HumanMessage(content="Test")],
                "session_id": f"test_{user_type}",
                "user_type": user_type,
                "processed": False
            }
            assert state["user_type"] == user_type
            
    def test_chat_state_modification(self):
        """Test ChatState field modification."""
        state: ChatState = {
            "messages": [HumanMessage(content="Initial")],
            "session_id": "mod_test",
            "user_type": "customer",
            "processed": False
        }
        
        # Test message addition
        state["messages"].append(AIMessage(content="Response"))
        assert len(state["messages"]) == 2
        
        # Test processed flag change
        state["processed"] = True
        assert state["processed"] is True
        
    def test_chat_state_empty_messages(self):
        """Test ChatState with empty messages list."""
        state: ChatState = {
            "messages": [],
            "session_id": "empty_test",
            "user_type": "customer",
            "processed": False
        }
        
        assert len(state["messages"]) == 0
        assert state["session_id"] == "empty_test" 