from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .services.property_api import PropertyAPI
from .services.image_processor import ImageProcessor, ImageAnalysisRequest
from .services.recommender import PropertyRecommender
from .models import UserPreferences, PropertySearchRequest, PropertySearchResponse, PropertyAnalysis, PropertyAnalysisResponse
from .llm_service import LLMService
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .services.property_scraper import PropertyScraper

app = FastAPI()

# Initialize services
# property_api = PropertyAPI(settings.DOMAIN_API_KEY)
image_processor = ImageProcessor()
recommender = PropertyRecommender()
property_scraper = PropertyScraper()

# Initialize LLM service without passing API key
llm_service = LLMService()

class ChatInput(BaseModel):
    user_input: str
    preferences: Optional[Dict] = None
    search_params: Optional[Dict] = None

# API Routers
# v1 API endpoints
v1_prefix = "/api/v1"

@app.post(f"{v1_prefix}/chat", tags=["Chat"])
async def chat_endpoint(chat_input: ChatInput):
    """Handle chat messages and preference updates"""
    print(chat_input)
    final_state = await llm_service.process_user_input(
        chat_input.user_input,
        chat_input.preferences,
        chat_input.search_params
    )

    response_data = {
        "response": final_state["messages"][-1].content if final_state["messages"] else "",
        "preferences": final_state["userpreferences"] if final_state["userpreferences"] else None,
        "search_params": final_state["propertysearchrequest"] if final_state["propertysearchrequest"] else None
    }

    # If we have search parameters, trigger the property search flow
    if final_state.get("is_complete"):
        try:
            # 1. Search for properties
            property_results = await search_properties(final_state["propertysearchrequest"])
            
            if property_results:
                # 2. Process images and create PropertyAnalysisResponse objects
                analyzed_properties: List[PropertyAnalysisResponse] = []
                
                for property in property_results:
                    if property.image_urls:
                        image_analysis = await process_image(
                            ImageAnalysisRequest(image_urls=property.image_urls)
                        )
                        # Create PropertyAnalysisResponse object
                        analyzed_property = PropertyAnalysisResponse(
                            properties_search=property,
                            image_analysis=image_analysis
                        )
                        analyzed_properties.append(analyzed_property)

                # 3. Get recommendations based on preferences and enriched properties
                if final_state.get("userpreferences") and analyzed_properties:
                    recommendations = await recommend_properties(
                        properties=analyzed_properties,
                        preferences=final_state["userpreferences"]
                    )
                    response_data["recommendations"] = recommendations

                response_data["properties"] = analyzed_properties

        except Exception as e:
            print(f"Error in property search flow: {str(e)}")
            # Don't raise exception, just continue with chat response
            pass

    return response_data

@app.post(f"{v1_prefix}/preferences", tags=["Preferences"])
async def update_preferences(preferences: UserPreferences):
    """Handle sidebar filter updates"""
    try:
        # Validate and store preferences
        return {
            "status": "success",
            "preferences": preferences.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{v1_prefix}/recommend", tags=["Recommendations"])
async def recommend_properties(
    properties: List[PropertyAnalysisResponse], 
    preferences: UserPreferences
) -> List[PropertyAnalysisResponse]:
    """Get property recommendations based on user preferences using LLM analysis"""
    try:
        recommendations = await recommender.get_recommendations(  
            properties=properties,
            preferences=preferences
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{v1_prefix}/process-image", tags=["Image Processing"])
async def process_image(image_urls: ImageAnalysisRequest) -> PropertyAnalysis:
    """Process a property image and return analysis results"""
    try:
        result = await image_processor.analyze_property_image(image_urls)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{v1_prefix}/properties/search", response_model=List[PropertySearchResponse], tags=["Properties"])
async def search_properties(search_params: PropertySearchRequest) -> List[PropertySearchResponse]:
    """
    Search for properties based on given criteria
    """
    try:
        results = await property_scraper.search_properties(
            location=search_params["location"],
            min_price=search_params["min_price"],
            max_price=search_params["max_price"],
            min_beds=search_params["min_bedrooms"],
            property_type=search_params["property_type"],
        )
        
        if not results:
            return []
            
        return [PropertySearchResponse(**result) for result in results]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching properties: {str(e)}"
        )