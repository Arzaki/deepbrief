"""Central configuration. All model names / budgets come from env, never hardcoded in logic."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = Field(default="groq")
    agent_model: str = Field(default="")
    cheap_model: str = Field(default="")
    judge_provider: str = Field(default="mock")
    judge_model: str = Field(default="")

    groq_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    tavily_api_key: str = ""

    temperature: float = 0.0
    seed: int = 42

    max_llm_calls: int = 30
    max_tool_calls: int = 30
    max_tokens: int = 60000
    max_wall_seconds: int = 600
    max_revisions: int = 2

    cache_enabled: bool = True
    cache_db: str = "./.cache/deepbrief.sqlite"
    trace_dir: str = "./traces"
    prompt_version: str = "v1"

    # Sensible per-provider defaults applied only when env leaves the model blank.
    PROVIDER_DEFAULTS: dict = {
        "groq": {"agent": "llama-3.3-70b-versatile", "cheap": "llama-3.1-8b-instant"},
        "openai": {"agent": "gpt-4o-mini", "cheap": "gpt-4o-mini"},
        "gemini": {"agent": "gemini-2.0-flash", "cheap": "gemini-2.0-flash-lite"},
        "mock": {"agent": "mock-agent", "cheap": "mock-cheap"},
    }

    def resolved_agent_model(self) -> str:
        if self.agent_model:
            return self.agent_model
        return self.PROVIDER_DEFAULTS.get(self.llm_provider, {}).get("agent", "mock-agent")

    def resolved_cheap_model(self) -> str:
        if self.cheap_model:
            return self.cheap_model
        return self.PROVIDER_DEFAULTS.get(self.llm_provider, {}).get("cheap", "mock-cheap")

    def resolved_judge_model(self) -> str:
        if self.judge_model:
            return self.judge_model
        # Judge should differ in family from generator when possible.
        if self.llm_provider != "gemini":
            return self.PROVIDER_DEFAULTS["gemini"]["agent"]
        return self.PROVIDER_DEFAULTS["openai"]["agent"]


def get_settings() -> Settings:
    return Settings()


# Price table (USD per 1K tokens). APPROXIMATE — for relative comparison only.
# Sources: public provider pricing pages, snapshot 2026-09. Update as needed.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.0-flash": {"input": 0.00010, "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    "mock-agent": {"input": 0.0, "output": 0.0},
    "mock-cheap": {"input": 0.0, "output": 0.0},
    "default": {"input": 0.0002, "output": 0.0006},
}

# Backwards-compat alias used by older imports.
DEFAULT_PRICE_TABLE = PRICE_TABLE


def get_price(model: str) -> dict[str, float]:
    return PRICE_TABLE.get(model, PRICE_TABLE["default"])
