"""
Minimal FastAPI application for the chatbot boilerplate.
Starting with a simple health check endpoint.
"""

import time
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import chat_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Chatbot Boilerplate",
        version="0.1.0",
        description="A comprehensive chatbot boilerplate with LangGraph and multi-user support",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(chat_routes.router, prefix="/api/v1")
    
    return app

# Create the app instance
app = create_app()

@app.get("/")
async def root():
    """Root endpoint with application information."""
    return {
        "name": "Chatbot Boilerplate",
        "version": "0.1.0",
        "status": "healthy",
        "docs_url": "/docs",
        "features": {
            "multi_user_support": "implemented",
            "vector_search": "planned", 
            "caching": "planned",
            "rate_limiting": "planned",
            "content_safety": "planned",
            "conversation_memory": "implemented",
            "dynamic_prompting": "implemented",
            "langgraph_integration": "implemented",
        },
        "endpoints": {
            "chat": "/api/v1/chat/",
            "chat_history": "/api/v1/chat/sessions/{session_id}/history",
            "chat_health": "/api/v1/chat/health"
        },
        "message": "🚀 Chatbot boilerplate is running! Visit /docs for API documentation."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "environment": "development",
        "services": {
            "api": "healthy",
            "llm": "configured" if "OPENAI_API_KEY" in os.environ else "not_configured",
            "vector_db": "not_configured",
            "cache": "not_configured",
            "chat_agent": "ready"
        },
        "uptime": "running"
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat": "/api/v1/chat (coming soon)",
            "admin": "/api/v1/admin (coming soon)",
            "data": "/api/v1/data (coming soon)"
        },
        "next_steps": [
            "Configure environment variables",
            "Set up LLM service",
            "Configure vector database",
            "Implement chat agent",
            "Add tool registry"
        ]
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
