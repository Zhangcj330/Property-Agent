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
    available_properties: Optional[List[FirestoreProperty]] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str
    response: str
    preferences: Optional[Dict] = None
    search_params: Optional[Dict] = None
    available_properties: Optional[List[FirestoreProperty]] = None
    latest_recommendation: Optional[PropertyRecommendationResponse] = None

class PropertyRecommendationRequest(BaseModel):
    """Request model for property recommendations based on state"""
    preferences: Dict
    search_params: Dict

# API Routers
# v1 API endpoints
v1_prefix = "/api/v1"

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
            
        # Save user's message
        user_message = ChatMessage(
            role="user",
            content=chat_input.user_input,
            timestamp=datetime.now()
        )
        await chat_storage.save_message(session_id, user_message)
        
        # Initialize the state with the user's message
        initial_state = {
            "messages": [HumanMessage(content=chat_input.user_input)],
            "session_id": session_id,
            "preferences": chat_input.preferences or {},
            "search_params": chat_input.search_params or {},
            "available_properties": chat_input.available_properties or [],
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
            available_properties=final_state["available_properties"],
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

@app.post(f"{v1_prefix}/saved_properties/", tags=["Properties"], status_code=201)
async def save_property_to_session(
    session_id: str,
    property_with_recommendation: PropertyWithRecommendation
):
    """Save a property with its recommendation to a session's saved properties list"""
    try:
        success = await firestore_service.save_property_to_session(
            session_id=session_id,
            property_with_recommendation=property_with_recommendation
        )
        return {
            "status": "success" if success else "error",
            "session_id": session_id,
            "property_id": property_with_recommendation.property.listing_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{v1_prefix}/saved_properties/{{session_id}}", tags=["Properties"])
async def get_saved_properties(session_id: str):
    """Get all saved properties with recommendations for a session"""
    try:
        properties = await firestore_service.get_saved_properties(session_id)
        return {
            "session_id": session_id,
            "properties": properties
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{v1_prefix}/saved_properties/{{session_id}}/{{property_id}}", tags=["Properties"])
async def remove_saved_property(
    session_id: str,
    property_id: str
):
    """Remove a property from a session's saved properties list"""
    try:
        success = await firestore_service.remove_saved_property(session_id, property_id)
        return {
            "status": "success" if success else "error",
            "session_id": session_id,
            "property_id": property_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{v1_prefix}/conversation/{{session_id}}", tags=["Agent"])
async def get_conversation_history(session_id: str):
    """Get all conversation messages for a session (for chat window rendering)"""
    try:
        chat_storage = ChatStorageService()
        session = await chat_storage.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # 返回消息列表，按时间排序
        messages = sorted(session.messages, key=lambda m: m.timestamp)
        return {
            "session_id": session_id,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error in get_conversation_history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching conversation: {str(e)}")