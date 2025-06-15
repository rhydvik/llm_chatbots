"""
Tests for chat utility functions.
"""

import pytest
from app.agents.utils.chat_utils import get_system_prompt


class TestChatUtils:
    """Test cases for chat utility functions."""
    
    def test_get_system_prompt_customer(self):
        """Test system prompt for customer user type."""
        prompt = get_system_prompt("customer")
        assert "helpful AI assistant for customers" in prompt
        assert "friendly" in prompt
        
    def test_get_system_prompt_support_agent(self):
        """Test system prompt for support agent user type."""
        prompt = get_system_prompt("support_agent")
        assert "AI assistant for support agents" in prompt
        assert "detailed, accurate information" in prompt
        
    def test_get_system_prompt_manager(self):
        """Test system prompt for manager user type."""
        prompt = get_system_prompt("manager")
        assert "AI assistant for managers" in prompt
        assert "strategic insights" in prompt
        assert "data-driven" in prompt
        
    def test_get_system_prompt_unknown_type(self):
        """Test system prompt for unknown user type defaults to customer."""
        prompt = get_system_prompt("unknown_type")
        customer_prompt = get_system_prompt("customer")
        assert prompt == customer_prompt
        
    def test_get_system_prompt_none_type(self):
        """Test system prompt for None user type."""
        prompt = get_system_prompt(None)
        customer_prompt = get_system_prompt("customer")
        assert prompt == customer_prompt
        
    def test_get_system_prompt_empty_string(self):
        """Test system prompt for empty string user type."""
        prompt = get_system_prompt("")
        customer_prompt = get_system_prompt("customer")
        assert prompt == customer_prompt
        
    def test_all_prompts_are_strings(self):
        """Test that all prompts return valid strings."""
        user_types = ["customer", "support_agent", "manager", "unknown"]
        for user_type in user_types:
            prompt = get_system_prompt(user_type)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            
    def test_prompts_are_different(self):
        """Test that different user types return different prompts."""
        customer_prompt = get_system_prompt("customer")
        support_prompt = get_system_prompt("support_agent")
        manager_prompt = get_system_prompt("manager")
        
        assert customer_prompt != support_prompt
        assert customer_prompt != manager_prompt
        assert support_prompt != manager_prompt 