"""
app/api/middleware/auth.py
───────────────────────────
Firebase Auth JWT middleware.
Validates Bearer tokens on every protected request.
Injects user_id into request state for downstream handlers.
"""

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

log = get_logger(__name__)

_firebase_initialized = False
_security = HTTPBearer(auto_error=False)


def _ensure_firebase_initialized() -> None:
    """
    Initialize Firebase Admin SDK (idempotent).
    Handles the case where Firebase is already initialized from a previous
    hot-reload cycle or multi-worker environment.
    """
    global _firebase_initialized
    if _firebase_initialized:
        return

    try:
        # Check if already initialized (e.g. by another import path)
        firebase_admin.get_app()
        _firebase_initialized = True
        return
    except ValueError:
        pass  # Not initialized yet — proceed

    try:
        if settings.firebase_service_account_path:
            cred = credentials.Certificate(settings.firebase_service_account_path)
            firebase_admin.initialize_app(cred)
        else:
            # Use application default credentials (works on GCP/Cloud Run)
            firebase_admin.initialize_app()
        _firebase_initialized = True
        log.info("firebase_initialized")
    except ValueError:
        # Already initialized (race condition in multi-threaded startup)
        _firebase_initialized = True


class FirebaseUser:
    """Decoded Firebase user injected into request state."""
    def __init__(self, uid: str, email: str | None = None, name: str | None = None):
        self.uid = uid
        self.email = email
        self.name = name

    def __repr__(self) -> str:
        return f"FirebaseUser(uid={self.uid!r}, email={self.email!r})"


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> FirebaseUser:
    """
    FastAPI dependency — extracts and verifies Firebase JWT.
    Usage: `user: FirebaseUser = Depends(get_current_user)`
    """
    _ensure_firebase_initialized()

    if credentials is None:
        raise AuthenticationError("No authorization header provided")

    token = credentials.credentials

    # Dev-mode bypass: allows testing without a real Firebase token
    if token == "dev-test-token" and not settings.is_production:
        log.debug("auth_verified_dev", uid="dev-test-user")
        return FirebaseUser(
            uid="dev-test-user",
            email="dev@example.com",
            name="Dev Test User",
        )

    try:
        decoded = firebase_auth.verify_id_token(token)
        user = FirebaseUser(
            uid=decoded["uid"],
            email=decoded.get("email"),
            name=decoded.get("name"),
        )
        log.debug("auth_verified", uid=user.uid)
        return user

    except firebase_auth.ExpiredIdTokenError:
        log.info("auth_token_expired", token_prefix=token[:8] + "...")
        raise AuthenticationError("Token has expired. Please sign in again.")
    except firebase_auth.InvalidIdTokenError:
        log.info("auth_token_invalid", token_prefix=token[:8] + "...")
        raise AuthenticationError("Invalid authentication token.")
    except firebase_auth.RevokedIdTokenError:
        log.info("auth_token_revoked", token_prefix=token[:8] + "...")
        raise AuthenticationError("Token has been revoked. Please sign in again.")
    except Exception as e:
        log.warning("auth_failed", error=str(e), error_type=type(e).__name__)
        raise AuthenticationError("Authentication failed.")

