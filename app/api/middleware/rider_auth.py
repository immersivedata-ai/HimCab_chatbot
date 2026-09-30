"""
app/api/middleware/rider_auth.py
─────────────────────────────────
Single-credential auth for the chat API.

The mobile app sends exactly ONE thing on every chat request:

    Authorization: Bearer <Him Cab rider JWT>

That same token does two jobs:
  1. Identifies the caller  → used as the session owner (`user_id`), and for the
     numeric `rider_id` the offer endpoints personalise on
  2. Authorizes backend calls → forwarded verbatim to the Him Cab fare and offer
     microservices

There is deliberately no second header, no rider id, and no session id — see
app/api/v1/routes/chat.py for the full front-end contract.

⚠️  SIGNATURE IS NOT VERIFIED HERE.
We do not hold Him Cab's JWT signing key, so this module DECODES the token to read
its claims but cannot prove the token is authentic. The real enforcement point is the
Him Cab fare service, which verifies the token itself and returns 401 for a forged or
expired one (so a fake token yields a chat session but never a real fare).

To make identity itself tamper-proof, ask the Java/Spring team for either:
  (a) the JWT signing public key / shared secret, or
  (b) a token-introspection endpoint we can call,
then verify inside `_decode_claims` before trusting `sub`.
"""

import base64
import binascii
import hashlib
import json
from typing import Any, Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

log = get_logger(__name__)

_security = HTTPBearer(auto_error=False)

# Dev-only token so the demo front-ends (and local curl testing) work without a real
# Him Cab JWT. Rejected in production.
DEV_TOKEN = "dev-test-token"
DEV_USER_ID = "dev-test-user"

# Claim names the Spring backend might use for the rider's stable identifier, in
# preference order. The first one present wins.
_ID_CLAIMS = ("riderId", "rider_id", "userId", "user_id", "sub", "id", "email")

#: Above this, a numeric claim is a phone number rather than a database id — an Indian
#: mobile number is ten digits starting at 6, and no autoincrement id on this backend
#: is anywhere near a billion. Used only to reject; see `_rider_id_from`.
_MAX_PLAUSIBLE_RIDER_ID = 1_000_000_000


class RiderIdentity:
    """The authenticated caller, derived from the single Bearer token.

    Attributes:
        user_id:  stable id used to own sessions in Firestore.
        jwt:      the raw token, forwarded to the fare and offer services. None in dev mode.
        claims:   decoded JWT payload (empty dict if the token isn't a readable JWT).
        rider_id: the rider's NUMERIC Spring id, when the token carries one. Needed to
                  personalise offers (`/offer/getOffersByTripType/{riderId}`); None
                  falls back to the unpersonalised catalogue rather than no offers.
    """

    def __init__(
        self,
        user_id: str,
        jwt: Optional[str],
        claims: dict[str, Any],
        rider_id: Optional[int] = None,
    ) -> None:
        self.user_id = user_id
        self.jwt = jwt
        self.claims = claims
        self.rider_id = rider_id

    def __repr__(self) -> str:
        # Never include the raw token — this shows up in logs and tracebacks.
        return (
            f"RiderIdentity(user_id={self.user_id!r}, rider_id={self.rider_id!r}, "
            f"has_jwt={bool(self.jwt)})"
        )


def _b64url_decode(segment: str) -> bytes:
    """Decode a base64url JWT segment, restoring the padding JWTs strip."""
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def _decode_claims(token: str) -> dict[str, Any]:
    """Read a JWT's payload claims WITHOUT verifying the signature.

    Returns {} for anything that isn't a readable JWT — callers must treat an empty
    dict as "identity unknown", never as "token valid".
    """
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _user_id_from(token: str, claims: dict[str, Any]) -> str:
    """Pick a stable session-owner id for this token.

    Prefers a real claim so the same rider keeps their history across app restarts and
    token refreshes. Falls back to a hash of the token itself, which still gives a
    consistent id for the life of that token instead of failing the request.
    """
    for claim in _ID_CLAIMS:
        value = claims.get(claim)
        if value not in (None, ""):
            return f"rider:{value}"

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
    log.warning("rider_token_without_id_claim", token_digest=digest[:8])
    return f"anon:{digest}"


def _rider_id_from(claims: dict[str, Any]) -> Optional[int]:
    """The rider's numeric Spring id, if the token carries one anywhere.

    Only the offer endpoints need this, and only to personalise: without it they fall
    back to the whole live catalogue. So an unparseable or absent claim is a normal
    outcome, not a failure — never raise, never guess.

    A claim like `sub: "9816000000"` is a phone number, not a rider id, so values are
    only accepted from id-shaped claims. `sub` is deliberately NOT one of them — and
    because a Spring backend might just as easily put the phone number in `userId`,
    anything large enough to BE a phone number is rejected too. Sending one as a
    riderId would silently fetch a different account's offers, which is worse than the
    unpersonalised catalogue we fall back to.
    """
    for claim in ("riderId", "rider_id", "userId", "user_id", "id"):
        value = claims.get(claim)
        if isinstance(value, bool) or value in (None, ""):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < parsed < _MAX_PLAUSIBLE_RIDER_ID:
            return parsed
    return None


def _normalize_bearer(token: str) -> str:
    """Strip a doubled 'Bearer ' prefix.

    An app that stores the header value (not the bare token) would otherwise send
    'Bearer Bearer <jwt>', which the fare service silently rejects as a 401.
    """
    cleaned = token.strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[len("bearer "):].strip()
    return cleaned


async def get_rider(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> RiderIdentity:
    """FastAPI dependency — the ONLY auth the chat endpoints require.

    Usage: `rider: RiderIdentity = Depends(get_rider)`
    """
    if credentials is None:
        raise AuthenticationError(
            "Missing Authorization header. Send 'Authorization: Bearer <rider JWT>'."
        )

    token = _normalize_bearer(credentials.credentials)
    if not token:
        raise AuthenticationError("Empty bearer token.")

    # Dev-mode bypass — keeps the demo front-ends and local testing working.
    if token == DEV_TOKEN:
        if settings.is_production:
            raise AuthenticationError("Invalid authentication token.")
        log.debug("rider_auth_dev_bypass", user_id=DEV_USER_ID)
        # jwt=None → the fare gate refuses to quote at all. NOT "degrades to
        # approximate pricing", which this comment used to claim and which does not
        # exist anywhere: there is no estimator, and there should not be — an invented
        # fare is exactly what `guardrails` is there to stop.
        #
        # The consequence is worth stating plainly, because it is the single most
        # common "the flow is broken" report: no JWT means no fare, and no fare means
        # no cab options, so the rider is offered no choice of vehicle. The three test
        # clients say so on screen when they are running on this token.
        return RiderIdentity(user_id=DEV_USER_ID, jwt=None, claims={})

    claims = _decode_claims(token)
    user_id = _user_id_from(token, claims)
    rider_id = _rider_id_from(claims)
    log.debug(
        "rider_auth_ok",
        user_id=user_id,
        has_claims=bool(claims),
        has_rider_id=rider_id is not None,
    )
    return RiderIdentity(user_id=user_id, jwt=token, claims=claims, rider_id=rider_id)
