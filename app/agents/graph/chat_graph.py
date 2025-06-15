"""
Chat agent graph builder using LangGraph.
"""

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..states.chat_state import ChatState
from ..nodes.chat_nodes import ChatNodes

logger = logging.getLogger(__name__)


class ChatGraphBuilder:
    """Builder for the chat agent's LangGraph workflow."""
    
    def __init__(self, chat_nodes: ChatNodes):
        self.chat_nodes = chat_nodes
    
    def build_graph(self, memory: MemorySaver):
        """Build the LangGraph state graph."""
        logger.info("Building chat agent graph...")
        
        workflow = StateGraph(ChatState)
        
        workflow.add_node("input_processing", self.chat_nodes.input_processing_node)
        workflow.add_node("llm_processing", self.chat_nodes.llm_processing_node)
        workflow.add_node("response_formatting", self.chat_nodes.response_formatting_node)
        
        workflow.add_edge(START, "input_processing")
        workflow.add_edge("input_processing", "llm_processing")
        workflow.add_edge("llm_processing", "response_formatting")
        workflow.add_edge("response_formatting", END)
        
        graph = workflow.compile(checkpointer=memory)
        
        logger.info("Chat agent graph built successfully")
        return graph 