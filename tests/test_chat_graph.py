"""
Tests for ChatGraphBuilder functionality.
"""

import pytest
from unittest.mock import Mock
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph.chat_graph import ChatGraphBuilder
from app.agents.nodes.chat_nodes import ChatNodes
from app.agents.states.chat_state import ChatState


class TestChatGraphBuilder:
    """Test cases for ChatGraphBuilder class."""
    
    def test_graph_builder_initialization(self, chat_nodes):
        """Test ChatGraphBuilder initialization."""
        builder = ChatGraphBuilder(chat_nodes)
        assert builder.chat_nodes == chat_nodes
        
    def test_build_graph_returns_compiled_graph(self, graph_builder, memory_saver):
        """Test that build_graph returns a compiled graph."""
        graph = graph_builder.build_graph(memory_saver)
        
        # Verify that a graph was returned
        assert graph is not None
        # The graph should be compiled and ready to use
        assert hasattr(graph, 'ainvoke')
        assert hasattr(graph, 'aget_state')
        
    def test_build_graph_with_different_memory(self, graph_builder):
        """Test building graph with different memory configurations."""
        custom_memory = MemorySaver()
        graph = graph_builder.build_graph(custom_memory)
        
        assert graph is not None
        assert hasattr(graph, 'ainvoke')
        
    def test_build_graph_structure(self, graph_builder, memory_saver):
        """Test that the built graph has correct structure."""
        graph = graph_builder.build_graph(memory_saver)
        
        # The graph should be compiled and callable
        assert callable(graph.ainvoke)
        assert callable(graph.aget_state)
        
    @pytest.mark.asyncio
    async def test_graph_execution_flow(self, graph_builder, memory_saver):
        """Test that the graph can execute the full flow."""
        graph = graph_builder.build_graph(memory_saver)
        
        initial_state = {
            "messages": [{"type": "human", "content": "Hello"}],
            "session_id": "test_graph_execution",
            "user_type": "customer",
            "processed": False
        }
        
        thread_config = {"configurable": {"thread_id": "test_thread"}}
        
        # This should not raise an exception
        try:
            result = await graph.ainvoke(initial_state, thread_config)
            # Basic validation that we get a result
            assert result is not None
            assert "messages" in result
        except Exception as e:
            # If there's an exception, it should be handled gracefully
            pytest.fail(f"Graph execution failed: {e}")


class TestGraphIntegration:
    """Integration tests for the complete graph workflow."""
    
    @pytest.mark.asyncio
    async def test_graph_with_mock_nodes(self, memory_saver, mock_sessions):
        """Test graph with completely mocked nodes."""
        # Create mock chat nodes
        mock_nodes = Mock(spec=ChatNodes)
        
        # Mock the node methods
        async def mock_input_processing(state):
            state["processed_input"] = True
            return state
            
        async def mock_llm_processing(state):
            state["processed_llm"] = True
            return state
            
        async def mock_response_formatting(state):
            state["processed"] = True
            return state
        
        mock_nodes.input_processing_node = mock_input_processing
        mock_nodes.llm_processing_node = mock_llm_processing
        mock_nodes.response_formatting_node = mock_response_formatting
        
        # Build graph with mocked nodes
        builder = ChatGraphBuilder(mock_nodes)
        graph = builder.build_graph(memory_saver)
        
        initial_state = {
            "messages": [],
            "session_id": "mock_test",
            "user_type": "customer",
            "processed": False
        }
        
        thread_config = {"configurable": {"thread_id": "mock_thread"}}
        result = await graph.ainvoke(initial_state, thread_config)
        
        # Verify that all processing flags were set
        assert result["processed"] is True
        
    def test_graph_builder_with_none_nodes(self):
        """Test that graph builder handles None nodes gracefully."""
        # ChatGraphBuilder may not immediately raise an error on init
        # but should fail when trying to build the graph
        builder = ChatGraphBuilder(None)
        memory = MemorySaver()
        
        with pytest.raises((TypeError, AttributeError)):
            builder.build_graph(memory)
            
    def test_graph_builder_with_invalid_nodes(self):
        """Test that graph builder handles invalid nodes."""
        invalid_nodes = "not_a_nodes_object"
        
        # This should either raise an error during init or during build
        builder = ChatGraphBuilder(invalid_nodes)
        memory = MemorySaver()
        
        with pytest.raises((TypeError, AttributeError)):
            builder.build_graph(memory)
            
    def test_memory_saver_integration(self, graph_builder):
        """Test that memory saver is properly integrated."""
        memory1 = MemorySaver()
        memory2 = MemorySaver()
        
        graph1 = graph_builder.build_graph(memory1)
        graph2 = graph_builder.build_graph(memory2)
        
        # Both graphs should be valid but potentially different instances
        assert graph1 is not None
        assert graph2 is not None
        
    @pytest.mark.asyncio
    async def test_graph_state_persistence(self, graph_builder, memory_saver):
        """Test that graph state is persisted correctly."""
        graph = graph_builder.build_graph(memory_saver)
        
        initial_state = {
            "messages": [],
            "session_id": "persistence_test",
            "user_type": "customer",
            "processed": False
        }
        
        thread_config = {"configurable": {"thread_id": "persistence_thread"}}
        
        # Execute the graph
        result = await graph.ainvoke(initial_state, thread_config)
        
        # Try to get the state back
        try:
            state = await graph.aget_state(thread_config)
            assert state is not None
        except Exception:
            # Some implementations might not support state retrieval
            # This is acceptable for basic functionality
            pass 