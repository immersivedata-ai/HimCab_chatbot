"""
app/services/himcab/
─────────────────────
Clients for the Him Cab Java/Spring backend (fare, category, vehicle, ...).

Distinct from `app/services/external/` (third-party APIs like weather): this package
talks to OUR own microservices at `settings.java_backend_url`, authenticated with the
rider's Spring JWT forwarded by the mobile app.
"""

from app.services.himcab.fare_service import FareService, get_fare_service
from app.services.himcab.models import FareQuote, FareQuoteInput
from app.services.himcab.quote_builder import build_fare_quote_input

__all__ = [
    "FareService",
    "get_fare_service",
    "FareQuote",
    "FareQuoteInput",
    "build_fare_quote_input",
]
