"""
Configuration management for the chatbot boilerplate.
Uses Pydantic Settings for environment variable handling and validation.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM-related configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    openai_api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4000, gt=0, description="Maximum tokens for LLM response")
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v or v == "your_openai_api_key_here":
            raise ValueError("OpenAI API key must be provided and valid")
        return v


class VectorDBSettings(BaseSettings):
    """Vector database configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    provider: str = Field(default="pinecone", description="Vector DB provider")
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: str = Field(default="chatbot-boilerplate-index", description="Pinecone index name")
    pinecone_dimension: int = Field(default=1536, description="Pinecone vector dimension")
    pinecone_metric: str = Field(default="cosine", description="Pinecone similarity metric")
    
    # Weaviate settings
    weaviate_url: Optional[str] = Field(default=None, description="Weaviate URL")
    weaviate_api_key: Optional[str] = Field(default=None, description="Weaviate API key")
    
    # Chroma settings
    chroma_host: Optional[str] = Field(default="localhost", description="Chroma host")
    chroma_port: Optional[int] = Field(default=8000, description="Chroma port")
    chroma_collection_name: str = Field(default="chatbot_collection", description="Chroma collection name")
    
    @validator('provider')
    def validate_provider(cls, v):
        allowed_providers = ["pinecone", "weaviate", "chroma"]
        if v not in allowed_providers:
            raise ValueError(f"Vector DB provider must be one of: {allowed_providers}")
        return v


class CacheSettings(BaseSettings):
    """Caching configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="CACHE_")
    
    backend: str = Field(default="memory", description="Cache backend type")
    ttl: int = Field(default=3600, gt=0, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, gt=0, description="Maximum cache size")
    
    # Redis settings
    redis_url: Optional[str] = Field(default=None, description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")
    
    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests: int = Field(default=100, gt=0, description="Number of requests allowed")
    window: int = Field(default=60, gt=0, description="Time window in seconds")
    storage: str = Field(default="memory", description="Rate limit storage backend")


class SecuritySettings(BaseSettings):
    """Security-related configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    secret_key: str = Field(..., description="Secret key for JWT and encryption")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, gt=0, description="Access token expiration")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="Allowed CORS methods"
    )
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v or v == "your_secret_key_here_change_in_production":
            raise ValueError("Secret key must be provided and changed from default")
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class GuardrailSettings(BaseSettings):
    """Content safety and guardrail settings."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    enable_content_guardrails: bool = Field(default=True, description="Enable content guardrails")
    max_input_length: int = Field(default=5000, gt=0, description="Maximum input length")
    max_output_length: int = Field(default=8000, gt=0, description="Maximum output length")
    blocked_words_file: Optional[str] = Field(default=None, description="Path to blocked words file")


class SessionSettings(BaseSettings):
    """Session management settings."""
    
    model_config = SettingsConfigDict(env_prefix="SESSION_")
    
    ttl: int = Field(default=86400, gt=0, description="Session TTL in seconds")
    max_sessions_per_user: int = Field(default=10, gt=0, description="Max sessions per user")
    cleanup_interval: int = Field(default=3600, gt=0, description="Session cleanup interval")


class FeatureFlags(BaseSettings):
    """Feature flag settings."""
    
    model_config = SettingsConfigDict(env_prefix="ENABLE_")
    
    auto_user_detection: bool = Field(default=True, description="Enable automatic user type detection")
    multi_stage_workflows: bool = Field(default=True, description="Enable multi-stage workflows")
    dynamic_prompting: bool = Field(default=True, description="Enable dynamic system prompting")
    conversation_memory: bool = Field(default=True, description="Enable conversation memory")
    tool_validation: bool = Field(default=True, description="Enable tool validation")
    metrics: bool = Field(default=True, description="Enable metrics collection")
    content_guardrails: bool = Field(default=True, description="Enable content guardrails")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Basic app settings
    app_name: str = Field(default="Chatbot Boilerplate", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, gt=0, le=65535, description="Server port")
    workers: int = Field(default=1, gt=0, description="Number of worker processes")
    
    # Database settings
    database_url: str = Field(default="sqlite:///./chatbot.db", description="Database URL")
    
    # Monitoring
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    health_check_endpoint: str = Field(default="/health", description="Health check endpoint")
    
    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    sessions: SessionSettings = Field(default_factory=SessionSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "echo": self.debug and self.is_development,
            "future": True
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.security.cors_origins,
            "allow_credentials": self.security.cors_allow_credentials,
            "allow_methods": self.security.cors_allow_methods,
            "allow_headers": self.security.cors_allow_headers
        }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings() 