from pydantic import BaseModel
from typing import List, Optional, Dict
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
    max_price: Optional[float] = None
    location: Optional[str] = None  # STATE-SUBURB-POSTCODE format
    suburb: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[int] = None
    min_bedrooms: Optional[int] = None
    property_type: Optional[str] = None
    must_have_features: List[str] = []
    demographic_analysis: Optional[Dict] = None
    suburb_recommendation_reason: Optional[str] = None 
