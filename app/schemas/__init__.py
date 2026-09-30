"""
app/schemas/
─────────────
Wire formats — the request and response bodies of the public HTTP API.

Separate from app/domain/ because the two change for different reasons and at
different costs. An entity can be refactored whenever the code needs it; a field in
here is a contract with a shipped mobile-app build, and renaming one breaks every
client that has not updated. Keeping them in one file made it easy to do the second
by accident while intending the first.

  chat.py     /v1/chat/* — including StructuredEvent, the card contract
  session.py  /v1/session/*
  alerts.py   /v1/alerts/himachal
"""
