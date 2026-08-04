"""
scripts/seed_data.py
─────────────────────
Seed Firestore with test data for local development.
Creates sample user profiles and trip sessions.

Usage:
    python scripts/seed_data.py
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta

# Ensure the app package is importable
sys.path.insert(0, ".")

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.models.session import (
    ExtractedEntities,
    Language,
    MessageRole,
    SessionStatus,
    TripSession,
)
from app.models.user import PastTrip, TravelPreferences, UserProfile
from app.services.firestore import get_firestore_client

configure_logging()
log = get_logger(__name__)


async def seed_users() -> None:
    """Create sample user profiles."""
    db = get_firestore_client()
    col = db.collection(settings.firestore_users_collection)

    users = [
        UserProfile(
            user_id="seed-user-outstation",
            firebase_uid="seed-user-outstation",
            display_name="Rahul Sharma",
            phone_number="+91-9876543210",
            home_city=None,
            preferences=TravelPreferences(
                preferred_budget="moderate",
                preferred_group_type="family",
                interests=["temples", "trekking", "local food"],
                dietary_restrictions=["vegetarian"],
                preferred_language=Language.HINDI,
            ),
            past_trips=[
                PastTrip(destination="Manali", duration_days=4, travel_date="2025-12-20"),
                PastTrip(destination="Shimla", duration_days=3, travel_date="2026-03-15"),
                PastTrip(destination="Dharamshala", duration_days=3, travel_date="2026-06-10"),
            ],
            total_trips=3,
        ),
        UserProfile(
            user_id="seed-user-local",
            firebase_uid="seed-user-local",
            display_name="Priya Thakur",
            phone_number="+91-9812345678",
            home_city="Shimla",
            preferences=TravelPreferences(
                preferred_budget="budget",
                preferred_group_type="friends",
                interests=["hidden gems", "cafes", "adventure"],
                preferred_language=Language.HINDI,
            ),
            past_trips=[
                PastTrip(destination="Kasol", duration_days=2, travel_date="2026-05-01"),
                PastTrip(destination="Bir Billing", duration_days=1, travel_date="2026-07-10"),
            ],
            total_trips=2,
        ),
        UserProfile(
            user_id="seed-user-premium",
            firebase_uid="seed-user-premium",
            display_name="Vikram Singh",
            phone_number="+91-9988776655",
            preferences=TravelPreferences(
                preferred_budget="luxury",
                preferred_group_type="couple",
                interests=["spa", "scenic drives", "fine dining"],
                preferred_language=Language.ENGLISH,
            ),
            past_trips=[
                PastTrip(destination="Manali", duration_days=5, travel_date="2025-06-15", rating=4.5),
                PastTrip(destination="Shimla", duration_days=4, travel_date="2025-09-20", rating=4.0),
                PastTrip(destination="Dharamshala", duration_days=3, travel_date="2025-12-01", rating=4.8),
                PastTrip(destination="Manali", duration_days=4, travel_date="2026-01-10", rating=4.2),
                PastTrip(destination="Kasol", duration_days=3, travel_date="2026-03-20", rating=4.6),
                PastTrip(destination="Spiti", duration_days=6, travel_date="2026-06-05", rating=5.0),
            ],
            total_trips=6,
            is_premium=True,
        ),
    ]

    for user in users:
        await col.document(user.user_id).set(user.to_firestore_dict())
        log.info("seeded_user", user_id=user.user_id, name=user.display_name)

    log.info("users_seeded", count=len(users))


async def main() -> None:
    log.info("seed_starting", env=settings.app_env)
    await seed_users()
    log.info("seed_complete")


if __name__ == "__main__":
    asyncio.run(main())
