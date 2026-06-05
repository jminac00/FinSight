import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["groq", "ollama", "openai"] = "openai"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-large"

    # External services
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    newsapi_key: str = ""
    finnhub_api_key: str = ""

    # Application
    frontend_url: str = "http://localhost:5173"
    environment: Literal["development", "production"] = "development"

    # Cache TTLs (seconds)
    cache_ttl_sentiment: int = 1800
    cache_ttl_fundamental: int = 86400
    cache_ttl_technical: int = 86400

    # Analysis parameters
    prediction_horizon_days: int = 10
    max_news_articles: int = 10
    graph_hop_depth: int = 2
    lru_cache_max_models: int = 10

    def validate_production_keys(self) -> None:
        """Raise if critical API keys are missing in production."""
        if self.environment != "production":
            return
        missing = []
        if not self.groq_api_key and self.llm_provider == "groq":
            missing.append("GROQ_API_KEY")
        if not self.openai_api_key and self.llm_provider == "openai":
            missing.append("OPENAI_API_KEY")
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.newsapi_key:
            missing.append("NEWSAPI_KEY")
        if not self.finnhub_api_key:
            missing.append("FINNHUB_API_KEY")
        if missing:
            raise ValueError(f"Missing required production env vars: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    settings = Settings()
    settings.validate_production_keys()
    return settings
