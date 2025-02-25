from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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

class UserPreferences(BaseModel):
    min_price: Optional[float] = None
    max_price: float
    location: str
    min_bedrooms: int
    property_type: Optional[str] = None
    must_have_features: List[str] = [] 