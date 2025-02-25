from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import get_db, engine
from .services.property_api import PropertyAPI
from .services.image_processor import ImageProcessor
from .services.recommender import PropertyRecommender
import os
from .models import Property, UserPreferences
from .llm_service import LLMService
import os
from typing import List
from .config import settings

# Create database tables

app = FastAPI()

# Initialize services
# property_api = PropertyAPI(settings.DOMAIN_API_KEY)
# image_processor = ImageProcessor()
# recommender = PropertyRecommender()

# Initialize LLM service without passing API key
llm_service = LLMService()

# Mock database
properties = [
    Property(
        id="1",
        address="123 Main St",
        city="San Francisco",
        state="CA",
        price=1200000,
        bedrooms=3,
        bathrooms=2,
        square_footage=1500,
        property_type="house",
        description="Beautiful home in prime location",
        image_url="https://example.com/image1.jpg"
    ),
    # Add more mock properties...
]

@app.post("/extract-preferences")
async def extract_preferences(user_input: str):
    try:
        response, preferences = await llm_service.process_user_input(
            user_input,
            # You could also pass chat history from the request if needed
        )
        return {
            "response": response,
            "preferences": preferences
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def recommend_properties(preferences: UserPreferences) -> List[Property]:
    matching_properties = []
    
    for property in properties:
        if (property.price <= preferences.max_price and 
            property.bedrooms >= preferences.min_bedrooms and 
            preferences.location.lower() in property.city.lower()):
            matching_properties.append(property)
    
    return matching_properties 

# @app.get("/sync-properties")
# async def sync_properties(db: Session = Depends(get_db)):
#     """Sync properties from external API to database"""
#     properties = await property_api.search_properties(location="San Francisco")
#     await property_api.sync_to_db(db, properties)
#     return {"message": "Properties synced successfully"}

# @app.get("/analyze-property-images/{property_id}")
# async def analyze_images(property_id: int, db: Session = Depends(get_db)):
#     """Analyze property images"""
#     property = db.query(db_models.DBProperty).filter(db_models.DBProperty.id == property_id).first()
#     if not property:
#         raise HTTPException(status_code=404, detail="Property not found")
    
#     analysis_results = []
#     for image_url in property.image_urls.split(","):
#         result = await image_processor.analyze_property_image(image_url)
#         analysis_results.append(result)
    
#     return analysis_results 