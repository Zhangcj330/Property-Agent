import aiohttp
import asyncio
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from ..models import Property
class PropertyAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Example using a real estate API (you'll need to replace with actual API endpoints)
        self.base_url = "https://api.realtor.com/v2"
        
    async def search_properties(
        self,
        location: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_beds: Optional[int] = None,
        property_type: Optional[str] = None
    ) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            params = {
                "location": location,
                "min_price": min_price,
                "max_price": max_price,
                "min_beds": min_beds,
                "property_type": property_type,
                "api_key": self.api_key
            }
            
            async with session.get(f"{self.base_url}/properties", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []

    async def sync_to_db(self, db: Session, properties: List[Dict]):
        """Sync properties from API to database"""
        for prop_data in properties:
            db_property = Property(
                external_id=prop_data["id"],
                address=prop_data["address"],
                city=prop_data["city"],
                state=prop_data["state"],
                price=prop_data["price"],
                bedrooms=prop_data["bedrooms"],
                bathrooms=prop_data["bathrooms"],
                square_footage=prop_data["square_footage"],
                property_type=prop_data["property_type"],
                description=prop_data["description"],
                image_urls=prop_data["image_urls"]
            )
            db.add(db_property)
        await db.commit() 