from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str          # only API key needed

    claude_model: str = "claude-sonnet-4-20250514"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
