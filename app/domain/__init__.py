"""
app/domain/
────────────
Entities and value objects — what the system IS, independent of how it is delivered.

Nothing in this package imports from app/api, app/schemas or app/integrations. It is
the innermost layer: a change to an HTTP response, a Firestore collection name or a
third-party provider must not reach in here.

  conversation.py  TripSession, Message, ExtractedEntities — the chat, as persisted
  itinerary.py     Itinerary, DayPlan, Activity, CabPackage — the generated plan
  user.py          UserProfile, past trips, loyalty tiers
  location.py      RiderLocation — a resolved GPS fix
  alerts.py        WeatherReport, TravelAlert
"""
