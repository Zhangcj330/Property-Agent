from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .services.property_api import PropertyAPI
from .services.image_processor import ImageProcessor, ImageAnalysisRequest,PropertyAnalysis
from .services.recommender import PropertyRecommender
from .models import UserPreferences, PropertySearchRequest, PropertySearchResponse, FirestoreProperty
from .llm_service import LLMService
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .services.property_scraper import PropertyScraper
from .services.firestore_service import FirestoreService

app = FastAPI()

# Initialize services
# property_api = PropertyAPI(settings.DOMAIN_API_KEY)
image_processor = ImageProcessor()
recommender = PropertyRecommender()
property_scraper = PropertyScraper()

# Initialize LLM service without passing API key
llm_service = LLMService()

# Initialize Firestore service
firestore_service = FirestoreService()

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

    # Extract response message content, handling both dict and object formats
    response_message = ""
    if final_state["messages"]:
        last_message = final_state["messages"][-1]
        if isinstance(last_message, dict) and "content" in last_message:
            response_message = last_message["content"]
        elif hasattr(last_message, "content"):
            response_message = last_message.content

    response_data = {
        "response": response_message,
        "preferences": final_state["userpreferences"] if final_state["userpreferences"] else None,
        "search_params": final_state["propertysearchrequest"] if final_state["propertysearchrequest"] else None
    }

    # If we have search parameters, trigger the property search flow
    if final_state.get("is_complete") :
        print("Search parameters found, triggering property search flow")
        try:
            # 1. Search for properties
            property_results = await search_properties(final_state["propertysearchrequest"])

            if property_results:
                analyzed_properties: List[FirestoreProperty] = []
                
                for property in property_results:
                    # Get or create property with analysis
                    stored_property = await firestore_service.get_property(property.listing_id)
                    
                    if stored_property and stored_property.analysis:
                        analyzed_properties.append(stored_property)
                    elif property.image_urls:
                    # Create new analysis
                        image_analysis = await process_image(
                            ImageAnalysisRequest(image_urls=property.image_urls)
                        )
                        # Save property and update with analysis
                        await firestore_service.save_property(property)
                        await firestore_service.update_property_analysis(
                            property.listing_id, 
                            image_analysis
                        )
                        # Get the updated property with analysis
                        analyzed_property = await firestore_service.get_property(property.listing_id)
                        if analyzed_property:
                            analyzed_properties.append(analyzed_property)

                # 3. Get recommendations based on preferences and enriched properties
                if final_state.get("userpreferences") and analyzed_properties:
                    recommendations = await recommend_properties(
                        properties=analyzed_properties,
                        preferences=final_state["userpreferences"]
                    )
                    print("Recommendations: ", recommendations)
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
    properties: List[FirestoreProperty], 
    preferences: UserPreferences
) -> List[FirestoreProperty]:
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

@app.post(f"{v1_prefix}/properties", tags=["Properties"], status_code=201)
async def create_property(property_data: PropertySearchResponse):
    """Save a new property listing"""
    try:
        listing_id = await firestore_service.save_property(property_data)
        return {"status": "success", "listing_id": listing_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{v1_prefix}/properties/{{listing_id}}", tags=["Properties"])
async def get_property(listing_id: str):
    """Get a property by listing ID"""
    try:
        property_data = await firestore_service.get_property(listing_id)
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        return property_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{v1_prefix}/properties/search", response_model=List[PropertySearchResponse], tags=["Properties"])
async def search_properties(search_params: PropertySearchRequest) -> List[PropertySearchResponse]:
    """Search for properties based on given criteria"""
    try:
        # First try to get from Firestore
        filters = {
            "location": search_params.get("location"),
            "suburb": search_params.get("suburb"),
            "state": search_params.get("state"),
            "postcode": search_params.get("postcode"),
            "min_price": search_params.get("min_price"),
            "max_price": search_params.get("max_price"),
            "min_bedrooms": search_params.get("min_bedrooms"),
            "property_type": search_params.get("property_type"),
        }
    
        stored_results = await firestore_service.list_properties(filters=filters)
        
        if stored_results:
            return stored_results
            
        # If no stored results, scrape new ones
        scraped_results = await property_scraper.search_properties(
            search_params=search_params,
            max_results=20
        )
        
        # Save scraped results to Firestore
        for result in scraped_results:
            await firestore_service.save_property(result)
            
        return scraped_results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching properties: {str(e)}"
        )