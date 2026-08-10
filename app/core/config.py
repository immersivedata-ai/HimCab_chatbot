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
    gemini_flash_model: str
    llm_call_timeout_seconds: float = 45.0    # hard cap so a stalled Vertex AI call can't hang a request forever

    # ── Firebase ──────────────────────────────────────────────────────────────
    firebase_project_id: str
    firebase_service_account_path: str = ""  # Optional on Cloud Run (uses ADC)

    # ── Firestore ─────────────────────────────────────────────────────────────
    firestore_database_id: str = "(default)"
    firestore_sessions_collection: str = "trip_sessions"
    firestore_users_collection: str = "user_profiles"
    # Durable "which conversation is this rider in" pointer — the system of record
    # for multi-turn chat.
    firestore_active_chats_collection: str = "active_chats"

    # ── Caching ───────────────────────────────────────────────────────────────
    # In-process only (see app/services/local_cache.py) — there is no cache server.
    weather_cache_ttl_seconds: int = 1800     # 30 min

    # ── External APIs ─────────────────────────────────────────────────────────
    openweather_api_key: str = ""
    java_backend_url: str = "http://34.131.37.50:9002"

    # ── Him Cab fare/cab services (Spring backend) ────────────────────────────
    # NOTE: the fare service is hosted at e.g. http://34.131.37.50:9002 — point
    # java_backend_url (or JAVA_BACKEND_URL env) at the fare host / API gateway.
    enable_real_fares: bool = True            # Feature flag; True = fetch real Him Cab fares
    himcab_fare_path: str = "/api/rider/fare/estimate"            # per-category fare estimate
    himcab_fare_getall_path: str = "/api/rider/fare/get-all"      # rate-card catalog (Phase 2)
    himcab_fare_timeout_seconds: float = 10.0

    # ── Him Cab offers (real coupons, same Spring backend) ────────────────────
    # The assistant quotes ONLY these — see app/services/himcab/offers_service.py for
    # why it fetches them rather than staying silent about discounts.
    enable_offers: bool = True                # Feature flag; False = never mention coupons
    #: Personalised, used whenever the rider's numeric id is known. `{riderId}` is
    #: substituted at call time.
    himcab_offers_by_trip_type_path: str = "/api/rider/offer/getOffersByTripType/{riderId}"
    #: Unpersonalised fallback for a JWT with no rider-id claim; filtered on our side.
    himcab_offers_all_path: str = "/api/rider/offer/get-all"
    himcab_offers_page_size: int = 50
    #: How many coupons the reply may name. More than a couple reads as a spam email
    #: rather than a recommendation; the app's offers screen has the full list.
    max_offers_in_reply: int = 3

    # ── Conversation policy ───────────────────────────────────────────────────
    # How the agent balances "ask for the missing detail" against "just answer".
    # These are the only knobs on that trade-off — the slot policy itself lives in
    # app/agent/slots.py.
    # How many times one detail may be asked for before the agent stops asking and
    # proceeds with a stated assumption instead. Also caps consecutive turns that
    # tell us nothing new. Set to 0 to never ask for anything optional — the
    # destination is still asked for, since nothing can be planned without it.
    max_clarification_attempts: int = 2
    # Trip length used when the user never gives one. Always disclosed in the reply
    # ("assuming a 3-day trip") — never presented as something the user said.
    default_trip_duration_days: int = 3

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    # Enforced in-process (app/services/local_cache.py), so this is the budget per
    # server instance rather than a global one.
    rate_limit_per_minute: int = 30

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

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
