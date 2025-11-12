"""
Configuration management for the Anthropic-to-OpenAI proxy.
Handles environment variables and settings.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Settings:
    """Application settings loaded from environment variables."""

    # Proxy Server Configuration
    PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "3000"))

    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # LangFuse Configuration (Optional)
    LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    LANGFUSE_API_KEY: Optional[str] = os.getenv("LANGFUSE_API_KEY")
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: Optional[str] = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # Model Mapping Configuration
    # Maps Anthropic model names to OpenRouter model identifiers
    MODEL_MAPPING: dict = {
        # Claude 3.5 Sonnet (Latest and recommended)
        "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",

        # Claude 3.5 Haiku
        "claude-3-5-haiku-20241022": "anthropic/claude-3.5-haiku",
        "claude-3.5-haiku": "anthropic/claude-3.5-haiku",

        # Claude 3 Opus
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-opus": "anthropic/claude-3-opus",

        # Claude 3 Sonnet
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        "claude-sonnet": "anthropic/claude-3-sonnet",

        # Claude 3 Haiku
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
        "claude-haiku": "anthropic/claude-3-haiku",

        # Claude 4.5 Haiku (New format)
        "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5",

        # Claude Opus 4 (New format)
        "claude-opus-4-20250514": "anthropic/claude-opus-4",

        # Claude Sonnet 4 (New format)
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",

        # Fallback: use model name as-is (formatted as anthropic/{name})
        # Any unrecognized claude-* model will be attempted with anthropic/ prefix
    }

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Set it in your .env file or export it in your shell."
            )

    @classmethod
    def get_openrouter_model(cls, anthropic_model: str) -> str:
        """
        Convert Anthropic model name to OpenRouter format.

        Args:
            anthropic_model: Model name from Anthropic API (e.g., 'claude-3-5-sonnet-20241022')

        Returns:
            OpenRouter model identifier (e.g., 'anthropic/claude-3.5-sonnet')
        """
        # Check if there's an environment variable override for this specific model
        override_key = f"MODEL_OVERRIDE_{anthropic_model.upper().replace('-', '_')}"
        if override_key in os.environ:
            override_model = os.environ[override_key]
            logger.info(f"Using model override: {anthropic_model} -> {override_model}")
            return override_model

        # Check if we have an exact mapping
        if anthropic_model in cls.MODEL_MAPPING:
            return cls.MODEL_MAPPING[anthropic_model]

        # If it's already in OpenRouter format (contains /), return as-is
        if "/" in anthropic_model:
            return anthropic_model

        # For claude-* models not in mapping, format as anthropic/{name}
        # This handles new models gracefully
        if anthropic_model.startswith("claude-"):
            formatted_model = f"anthropic/{anthropic_model}"
            logger.debug(f"Using fallback format for unknown Claude model: {anthropic_model} -> {formatted_model}")
            return formatted_model

        # Fallback: assume it's a claude model
        return f"anthropic/{anthropic_model}"


# Create singleton settings instance
settings = Settings()
