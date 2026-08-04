import httpx
from app.core.config import settings
from app.core.logging import get_logger
from app.models.booking import TripDTORequest

log = get_logger(__name__)

async def book_trip_api(trip_request: TripDTORequest) -> dict:
    """
    Sends the trip booking request to the Java Spring Boot backend.
    """
    url = f"{settings.java_backend_url.rstrip('/')}/api/v1/rider/trip/book-ride"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # "Authorization": f"Bearer {auth_token}" # Would be injected here in prod
    }
    
    log.info("booking_api_call", url=url, payload=trip_request.model_dump())
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, json=trip_request.model_dump(), headers=headers)
            
            # The backend returns a BaseDTO which has a "response" field if successful
            response.raise_for_status()
            
            data = response.json()
            log.info("booking_api_success", response=data)
            return data
            
    except httpx.HTTPStatusError as e:
        log.error("booking_api_status_error", status_code=e.response.status_code, response=e.response.text)
        return {"error": f"Failed to book trip: {e.response.text}"}
    except Exception as e:
        log.error("booking_api_error", error=str(e))
        return {"error": "Failed to connect to the booking server."}
