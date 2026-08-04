from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 4
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # ── GCP ──────────────────────────────────────────────────────────────────
    gcp_project_id: str
    gcp_region: str
    google_application_credentials: str = ""  # Optional on Cloud Run (uses ADC)

    # ── Vertex AI / Gemini ────────────────────────────────────────────────────
    vertex_ai_location: str
    gemini_pro_model: str
    gemini_flash_model: str
    gemini_max_output_tokens: int = 8192
    gemini_temperature: float = 0.7

    # ── Firebase ──────────────────────────────────────────────────────────────
    firebase_project_id: str
    firebase_service_account_path: str = ""  # Optional on Cloud Run (uses ADC)

    # ── Firestore ─────────────────────────────────────────────────────────────
    firestore_database_id: str = "(default)"
    firestore_sessions_collection: str = "trip_sessions"
    firestore_users_collection: str = "user_profiles"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_ttl_weather: int = 1800      # 30 min
    redis_ttl_places: int = 86400      # 24 h
    redis_ttl_session: int = 7200      # 2 h

    # ── External APIs ─────────────────────────────────────────────────────────
    openweather_api_key: str = ""
    java_backend_url: str = "http://localhost:8080"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 10

    # ── Pub/Sub ───────────────────────────────────────────────────────────────
    pubsub_alert_topic: str = "himachal-travel-alerts"
    pubsub_alert_subscription: str = "alert-consumer-sub"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton. Use as a FastAPI dependency."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()

# Inject GCP project ID into environment for google-auth library
import os
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    os.environ["GOOGLE_CLOUD_PROJECT"] = settings.gcp_project_id
