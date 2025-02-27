from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from typing_extensions import TypedDict

class Property(BaseModel):
    id: str
    address: str
    city: str
    state: str
    price: float
    bedrooms: int
    bathrooms: float
    square_footage: float
    property_type: str
    description: str
    image_url: Optional[str] = None

class UserPreferences(TypedDict):
    Location: Tuple[Optional[str], float] = (None, 1.0)
    Price: Tuple[Optional[str], float] = (None, 1.0)
    Size: Tuple[Optional[str], float] = (None, 1.0)
    Layout: Tuple[Optional[str], float] = (None, 1.0)
    PropertyType: Tuple[Optional[str], float] = (None, 1.0)
    Features: Tuple[Optional[str], float] = (None, 1.0)
    Condition: Tuple[Optional[str], float] = (None, 1.0)
    Environment: Tuple[Optional[str], float] = (None, 1.0)
    Style: Tuple[Optional[str], float] = (None, 1.0)
    Quality: Tuple[Optional[str], float] = (None, 1.0)
    Room: Tuple[Optional[str], float] = (None, 1.0)
    SchoolDistrict: Tuple[Optional[str], float] = (None, 1.0)
    Community: Tuple[Optional[str], float] = (None, 1.0)
    Transport: Tuple[Optional[str], float] = (None, 1.0)
    Other: Tuple[Optional[str], float] = (None, 1.0)

# Request model for property search
class PropertySearchRequest(TypedDict):
    location: Optional[str] = Field(None, description="Location")
    suburb: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[int] = None
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")
    min_bedrooms: Optional[int] = Field(None, description="Minimum number of bedrooms")
    property_type: Optional[str] = Field(None, description="Type of property")

# Response model for property search
class PropertySearchResponse(BaseModel):
    listing_id: Optional[str]
    price: str
    address: str
    bedrooms: str
    bathrooms: str
    car_parks: Optional[str]
    land_size: Optional[str]
    property_type: Optional[str]
    inspection_date: Optional[str]
    image_urls: Optional[List[str]]
    agent_name: Optional[str]