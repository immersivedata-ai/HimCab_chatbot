"""
app/integrations/himcab/
─────────────────────
Clients for the Him Cab Java/Spring backend (fare, category, vehicle, ...).

Distinct from `app/integrations/external/` (third-party APIs like weather): this package
talks to OUR own microservices at `settings.java_backend_url`, authenticated with the
rider's Spring JWT forwarded by the mobile app.
"""

from app.integrations.himcab.fare_service import FareService, get_fare_service
from app.integrations.himcab.models import FareQuote, FareQuoteInput, Offer
from app.integrations.himcab.offers_service import (
    OffersService,
    get_offers_service,
    usable_offers,
)
from app.integrations.himcab.quote_builder import (
    build_fare_quote_input,
    build_rental_quote_input,
    matching_package,
)
from app.integrations.himcab.rental_service import (
    RentalPackage,
    available_packages,
    default_packages,
    get_rental_service,
    offered_hours,
    offered_packages,
    packages_payload,
)

__all__ = [
    "FareQuote",
    "FareQuoteInput",
    "FareService",
    "Offer",
    "OffersService",
    "RentalPackage",
    "available_packages",
    "build_fare_quote_input",
    "build_rental_quote_input",
    "default_packages",
    "get_fare_service",
    "get_offers_service",
    "get_rental_service",
    "matching_package",
    "offered_hours",
    "offered_packages",
    "packages_payload",
    "usable_offers",
]
