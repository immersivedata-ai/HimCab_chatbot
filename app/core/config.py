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

    # ── Staying inside Vertex's shared quota ──────────────────────────────────
    #
    # Gemini on Vertex runs on DYNAMIC SHARED QUOTA: capacity is a pool, a 429
    # `ResourceExhausted` means the pool is busy, and there is no quota increase to
    # request — that is the documented answer. The only levers on this side are sending
    # fewer requests and spreading them out. These three are those levers.
    #
    # Production hit sustained 429s. The arithmetic behind them: 3 model calls per turn,
    # up to 800 concurrent turns (10 instances x 80 concurrency), and LangChain silently
    # retrying each call six times — a pathological turn was 30 requests, and nothing
    # anywhere bounded the total.

    #: LangChain's own retry, which is invisible: `ChatVertexAI` defaults to 6 and the
    #: only sign is a `Retrying ... ResourceExhausted` warning. Six retries into an
    #: exhausted pool is six more requests that will also fail, so this is deliberately
    #: low. Backoff is still exponential; we are shortening the ladder, not removing it.
    vertex_max_retries: int = 2

    #: Token-bucket pacing applied to every model call, PER PROCESS. With
    #: `--max-instances=10` the fleet rate is ten times this, so treat it as a per-
    #: instance brake rather than a global guarantee — a true global cap belongs at the
    #: gateway, or in the instance ceiling. Same caveat as the rider rate limiter in
    #: `local_cache.py`.
    vertex_requests_per_second: float = 4.0
    #: How much of a burst the bucket will absorb before pacing bites. A chat turn fires
    #: its calls back to back, so a bucket smaller than one turn would add latency to
    #: every single turn for no benefit.
    vertex_burst_size: float = 8.0

    #: Hard ceiling on concurrent in-flight Vertex calls per process. The rate limiter
    #: paces STARTS; this bounds how many can be outstanding at once when the service is
    #: slow, which is exactly when a burst turns into a thundering herd.
    vertex_max_in_flight: int = 8

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
    #: stored in this repo — see app/integrations/himcab/rental_service.py.
    rental_packages_cache_ttl_seconds: int = 3_600      # 1 h
    #: THE durations the rental product sells. An ALLOWLIST, not a preference: a rider
    #: is offered these and nothing else, whatever charts exist in the Spring backend.
    #:
    #: The rate card is still the source of truth for what can be PRICED and for what
    #: each package INCLUDES — it supplies the kilometres against these hours. What it
    #: cannot do is introduce a duration of its own: a 6- or 12-hour chart left
    #: configured in the backend must not surface as a chip on a screen that sells 4
    #: and 8. This is also the list the assistant offers when the rate card cannot be
    #: read at all, so the rider is never asked an open "how many hours?" that invites
    #: an answer the backend matches no chart for.
    #:
    #: One list, deliberately. There used to be a separate min/max window alongside it,
    #: which meant two settings could disagree about the same product statement — and
    #: the one that lost was whichever the reader did not check. Anything wanting a
    #: range takes min()/max() of this.
    #:
    #: Env: RENTAL_HOUR_OPTIONS="4,8".
    rental_hour_options: list[int] = Field(default=[4, 8])

    # ── Him Cab offers (real coupons, same Spring backend) ────────────────────
    # The assistant quotes ONLY these — see app/integrations/himcab/offers_service.py for
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
            v = list(v.replace("[", "").replace("]", "").split(","))
        hours = {int(str(h).strip()) for h in v if str(h).strip()}
        return sorted(h for h in hours if h > 0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton. Use as a FastAPI dependency."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()

# Inject the GCP project id into the environment for the google-auth library.
# Deliberately at the bottom: it needs `settings`, which is constructed above, so
# it cannot move to the top of the file with the other imports.
import os  # noqa: E402

if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    os.environ["GOOGLE_CLOUD_PROJECT"] = settings.gcp_project_id
