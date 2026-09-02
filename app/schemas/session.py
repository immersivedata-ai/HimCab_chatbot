"""
app/schemas/session.py
───────────────────────
Request/response bodies for /v1/session/*.

Wire formats only. The entity these describe is `app.domain.conversation.TripSession`.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.domain.conversation import Language


class CreateSessionRequest(BaseModel):
    language: Language = Language.ENGLISH
    initial_message: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    status: str
