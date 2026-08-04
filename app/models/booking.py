from typing import List, Optional
from pydantic import BaseModel, Field

class DirectionDTO(BaseModel):
    address: str
    latitude: Optional[str] = None
    longitude: Optional[str] = None

class TripDTORequest(BaseModel):
    categoryId: int = 1  # Default to 1 (e.g., standard cab)
    fromDirectionAddress: DirectionDTO
    toDirectionAddress: List[DirectionDTO]
    date: str  # YYYY-MM-DD
    bookingTime: str  # HH:mm:ss
    distance: float = 0.0
    tripType: str  # LOCAL, RENTAL, OUTSTATION
    riderId: int = 1  # Mocked user ID
    tripStatus: str = "UPCOMING"
    
    # Optional fields for full compatibility
    tripRating: Optional[float] = None
    offerId: Optional[int] = None
