from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .services.property_api import PropertyAPI
from .services.image_processor import ImageProcessor, ImageAnalysisRequest,PropertyAnalysis
from .services.recommender import PropertyRecommender
from .models import UserPreferences, PropertySearchRequest, PropertySearchResponse, FirestoreProperty, PropertyRecommendationResponse, PropertyWithRecommendation, PropertyRecommendationInfo
from .llm_service import LLMService
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from .services.property_scraper import PropertyScraper
from .services.firestore_service import FirestoreService
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ChatMessage
from .Agent.agent import get_session_state, agent
from .services.chat_storage import ChatStorageService
from datetime import datetime

import uuid
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源 - 生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有头部
)

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
    session_id: Optional[str] = None  # 确保默认值为 None
    user_input: str = Field(..., min_length=1)  # 确保用户输入不为空
    preferences: Optional[Dict] = {}  # 设置默认空字典
    search_params: Optional[Dict] = {}  # 设置默认空字典
    recommendation_history: Optional[List[str]] = None
    latest_recommendation: Optional[PropertyRecommendationResponse] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str
    response: str
    preferences: Optional[Dict] = None
    search_params: Optional[Dict] = None
    recommendation_history: Optional[List[str]] = None
    latest_recommendation: Optional[PropertyRecommendationResponse] = None

class PropertyRecommendationRequest(BaseModel):
    """Request model for property recommendations based on state"""
    preferences: Dict
    search_params: Dict

# API Routers
# v1 API endpoints
v1_prefix = "/api/v1"

@app.post(f"{v1_prefix}/chat", tags=["Chat"], response_model=ChatResponse)
async def chat_endpoint(chat_input: ChatInput):
    """处理聊天消息并返回 AI 状态"""
    try:
        logger.info(f"Received chat input: {chat_input}")
        
        # 确保 session_id 为 None 或字符串
        session_id = chat_input.session_id if chat_input.session_id else str(uuid.uuid4())
        logger.info(f"Using session_id: {session_id}")
        
        # 确保 preferences 和 search_params 是字典
        preferences = chat_input.preferences or {}
        search_params = chat_input.search_params or {}
        
        logger.info(f"Processing user input with session_id: {session_id}")
        final_state = await llm_service.process_user_input(
            session_id=session_id,
            user_input=chat_input.user_input,
            preferences=preferences,
            search_params=search_params
        )
        logger.info("Successfully processed user input")
        
        # 构建响应
        response = ChatResponse(
            session_id=session_id,
            response=final_state["messages"][-1].content if final_state["messages"] else "",
            preferences=final_state["userpreferences"],
            search_params=final_state["propertysearchrequest"],
            is_complete=final_state.get("is_complete", False)
        )
        logger.info(f"Returning response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@app.post(f"{v1_prefix}/chat/recommend", tags=["Chat"], response_model=PropertyRecommendationResponse)
async def chat_recommendation_endpoint(request: PropertyRecommendationRequest):
    """Get property recommendations based on chat state"""
    try:
        # Search for properties based on the search parameters
        property_results = await search_properties(request.search_params)
        
        if not property_results:
            return PropertyRecommendationResponse(properties=[])

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

        # Get recommendations based on preferences and enriched properties
        if request.preferences and analyzed_properties:
            recommendations = await recommend_properties(
                properties=analyzed_properties,
                preferences=request.preferences
            )
            return recommendations
        
        return PropertyRecommendationResponse(properties=[])

    except Exception as e:
        print(f"Error in recommendation process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
) -> PropertyRecommendationResponse:
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

@app.post(f"{v1_prefix}/agent/chat", tags=["Agent"], response_model=ChatResponse)
async def agent_chat_endpoint(chat_input: ChatInput):
    """Process chat messages using the LangGraph agent"""
    try:
        logger.info(f"Received agent chat input: {chat_input}")
        
        # Ensure session_id exists
        session_id = chat_input.session_id if chat_input.session_id else str(uuid.uuid4())
        logger.info(f"Using agent session_id: {session_id}")
        # Ensure session exists
        chat_storage = ChatStorageService()

        session = await chat_storage.get_session(session_id)
        if not session:
            session = await chat_storage.create_session(session_id)
        
        # Initialize the state with the user's message
        initial_state = {
            "messages": [HumanMessage(content=chat_input.user_input)],
            "session_id": session_id,
            "preferences": chat_input.preferences or {},
            "search_params": chat_input.search_params or {},
            "recommendation_history": chat_input.recommendation_history or [],
            "latest_recommendation": None
        }

        # Run the agent
        final_state = await agent.ainvoke(initial_state)
        logger.info("Successfully processed input through agent")
        
        if final_state["messages"]:
            assistant_message = ChatMessage(
                role="assistant",
                content=final_state["messages"][-1].content,
                timestamp=datetime.now()
            )
            await chat_storage.save_message(session_id, assistant_message)
        
        # Extract the last message and any updated state
        last_message = final_state["messages"][-1].content if final_state["messages"] else ""
        
        # Construct response
        response = ChatResponse(
            session_id=session_id,
            response=last_message,
            preferences=final_state["preferences"],
            search_params=final_state["search_params"],
            recommendation_history=final_state["recommendation_history"],
            latest_recommendation=final_state["latest_recommendation"]
        )
        
        logger.info(f"Returning agent response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in agent_chat_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )