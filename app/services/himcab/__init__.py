"""
app/services/himcab/
─────────────────────
Clients for the Him Cab Java/Spring backend (fare, category, vehicle, ...).

Distinct from `app/services/external/` (third-party APIs like weather): this package
talks to OUR own microservices at `settings.java_backend_url`, authenticated with the
rider's Spring JWT forwarded by the mobile app.
"""

from app.services.himcab.fare_service import FareService, get_fare_service
from app.services.himcab.models import FareQuote, FareQuoteInput, Offer
from app.services.himcab.offers_service import (
    OffersService,
    get_offers_service,
    usable_offers,
)
from app.services.himcab.quote_builder import (
    build_fare_quote_input,
    build_rental_quote_input,
    matching_package,
)
from app.services.himcab.rental_service import (
    RentalPackage,
    available_packages,
    default_packages,
    get_rental_service,
    offered_hours,
    offered_packages,
    packages_payload,
)

__all__ = [
    "FareService",
    "get_fare_service",
    "FareQuote",
    "FareQuoteInput",
    "Offer",
    "OffersService",
    "get_offers_service",
    "usable_offers",
    "build_fare_quote_input",
    "build_rental_quote_input",
    "matching_package",
    "RentalPackage",
    "available_packages",
    "default_packages",
    "get_rental_service",
    "offered_hours",
    "offered_packages",
    "packages_payload",
]
