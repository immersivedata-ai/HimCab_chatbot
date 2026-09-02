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

    # ── Geocoding (places ↔ coordinates, live) ────────────────────────────────
    # Nominatim-compatible. There are NO coordinates stored in this codebase: every
    # place a rider names is resolved here, and which state it sits in comes from the
    # provider's administrative data. Point this at a self-hosted instance for
    # production — the public server is rate-limited and asks to be used lightly.
    geocoding_base_url: str = "https://nominatim.openstreetmap.org"
    #: Nominatim rejects anonymous traffic. Identify the deployment, with contact info.
    geocoding_user_agent: str = "HimCabTravelAssistant/1.0 (support@himcab.example)"
    #: Scopes the search so "Manali" cannot match a hamlet on another continent.
    geocoding_country_codes: str = "in"
    #: A town does not move between two messages, and the provider asks for at most one
    #: request a second — so results are held briefly rather than re-fetched per turn.
    geocoding_cache_ttl_seconds: int = 86_400     # 24 h

    # ── Routing (real road distance + drive time, live) ────────────────────────
    # OSRM-compatible `/route/v1` shape. Replaces a table of surveyed routes and the
    # great-circle fudge factors that filled its gaps.
    routing_base_url: str = "https://router.project-osrm.org"
    routing_path: str = "/route/v1/driving"
    routing_cache_ttl_seconds: int = 21_600       # 6 h

    # ── Him Cab fare/cab services (Spring backend) ────────────────────────────
    # NOTE: the fare service is hosted at e.g. http://34.131.37.50:9002 — point
    # java_backend_url (or JAVA_BACKEND_URL env) at the fare host / API gateway.
    enable_real_fares: bool = True            # Feature flag; True = fetch real Him Cab fares
    himcab_fare_path: str = "/api/rider/fare/estimate"            # per-category fare estimate
    himcab_fare_getall_path: str = "/api/rider/fare/get-all"      # rate card, incl. rental packages
    himcab_fare_timeout_seconds: float = 10.0

    # ── Him Cab rentals (car by the hour, same Spring backend) ────────────────
    enable_rentals: bool = True                # Feature flag; False = no rental route
    #: The rate card changes rarely and every rental turn needs it, so it is held
    #: briefly rather than re-fetched per message. Packages themselves are NEVER
    #: stored in this repo — see app/services/himcab/rental_service.py.
    rental_packages_cache_ttl_seconds: int = 3_600      # 1 h
    #: The durations the rental PRODUCT sells, as the app's Rent a Car screen states
    #: them: "you can rent a car for max 8hrs and min 3hrs for the day". The rate card
    #: is still the source of truth for what can be PRICED — these three only decide
    #: what the rider is offered, and are what the assistant falls back to when the
    #: rate card cannot be read (a duration from product config beats an open question
    #: the rider can only answer wrongly). Env: RENTAL_HOUR_OPTIONS="3,5,8".
    rental_hour_options: list[int] = Field(default=[3, 5, 8])
    #: The bookable window. A rental chart outside it is NOT offered even when the rate
    #: card defines one — a 12-hour chart left in the backend must not reappear as a
    #: chip on a screen that says the maximum is 8.
    rental_min_hours: int = 3
    rental_max_hours: int = 8

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

    @field_validator("rental_hour_options", mode="before")
    @classmethod
    def parse_rental_hours(cls, v: str | list) -> list[int]:
        """Accept "3,5,8" as well as a JSON list, and always hand back it sorted.

        The order IS the order the rider sees the options in, so it is normalised here
        rather than left to whoever set the env var.
        """
        if isinstance(v, str):
            v = [part for part in v.replace("[", "").replace("]", "").split(",")]
        hours = {int(str(h).strip()) for h in v if str(h).strip()}
        return sorted(h for h in hours if h > 0)


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
