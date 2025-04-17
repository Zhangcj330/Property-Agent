import os
import sys
from pathlib import Path
import asyncio

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

from typing_extensions import Literal
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END

from app.config import settings
from app.models import (
    PropertySearchRequest,
    UserPreferences,
    UserPreference,
    FirestoreProperty,
    ChatSession,
    ChatMessage,
    PropertyRecommendationResponse,
    PropertySearchResponse
)
from app.services.image_processor import ImageProcessor, ImageAnalysisRequest, PropertyAnalysis
from app.services.property_scraper import PropertyScraper
from app.services.recommender import PropertyRecommender
from app.services.chat_storage import ChatStorageService
from app.services.firestore_service import FirestoreService
from app.services.preference_service import (
    PreferenceService,
)
from app.services.planning_service import get_planning_info
from app.services.investment_service import InvestmentService
from IPython.display import Image, display
from langchain_community.tools import DuckDuckGoSearchResults
from app.services.sql_service import SQLService

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

# Initialize services
image_processor = ImageProcessor()
property_scraper = PropertyScraper()
recommender = PropertyRecommender()
chat_storage = ChatStorageService()
firestore_service = FirestoreService()
investment_service = InvestmentService()
sql_service = SQLService()  # Initialize SQL service

# Default LLM
llm = ChatGoogleGenerativeAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash",
)

# Define custom state class
class AgentMessagesState(MessagesState):
    """State class for the agent that extends MessagesState with additional fields"""
    session_id: str
    preferences: Dict[str, Any]
    search_params: Dict[str, Any]
    recommendation_history: List[str] = Field(default_factory=list, description="List of listing_ids that have been recommended")
    latest_recommendation: Optional[PropertyRecommendationResponse] = Field(default=None, description="Most recent property recommendations")
    available_properties: List[FirestoreProperty] = Field(default_factory=list, description="List of properties from the latest search")

# Tool parameter models
class ExtractPreferencesInput(BaseModel):
    session_id: str
    user_message: str

class HandleRejectionInput(BaseModel):
    session_id: str
    rejection_message: str
    property_details: Dict[str, Any]

class GetSessionStateInput(BaseModel):
    session_id: str

@tool
async def get_session_state(session_id: str) -> dict:
    """Get current preferences and search parameters
    
    Args:
        session_id: str - The ID of the chat session
    
    Returns:
        dict: A dictionary containing the current preferences and search parameters
    """
    service = PreferenceService()
    session = await service.chat_storage.get_session(session_id)
    if not session:
        return {"preferences": {}, "search_params": {}}
    
    return {
        "preferences": session.preferences or {},
        "search_params": session.search_params or {}
    }

async def process_property(result: PropertySearchResponse) -> FirestoreProperty:
    # Convert to FirestoreProperty
    firestore_property = FirestoreProperty.from_search_response(result)
    
    # Check if property exists in Firestore with analysis
    stored_property = await firestore_service.get_property(firestore_property.listing_id)
    
    if stored_property and stored_property.analysis:
        # Use existing analysis if available
        return stored_property
    
    # Initialize tasks list
    tasks = []
    
    # Add planning info task if address available
    if firestore_property.basic_info.full_address:
        async def get_planning():
            try:
                planning_info = await get_planning_info(firestore_property.basic_info.full_address)
                if not planning_info.error:
                    firestore_property.planning_info = planning_info
            except Exception as e:
                print(f"Error getting planning info for {firestore_property.listing_id}: {str(e)}")
        tasks.append(get_planning())
    
    # Add investment metrics task if suburb info available
    if firestore_property.basic_info.suburb and firestore_property.basic_info.postcode:
        async def get_investment():
            try:
                investment_metrics = investment_service.get_investment_metrics(
                    suburb=firestore_property.basic_info.suburb,
                    postcode=firestore_property.basic_info.postcode,
                    bedrooms=firestore_property.basic_info.bedrooms_count or 2
                )
                firestore_property.investment_info = investment_metrics
            except Exception as e:
                print(f"Error getting investment metrics for {firestore_property.listing_id}: {str(e)}")
        tasks.append(get_investment())
    
    # Add image analysis task if images available
    if firestore_property.media.image_urls:
        async def analyze_images():
            try:
                # Create tasks for parallel execution
                analysis_task = image_processor.analyze_property_image(
                    ImageAnalysisRequest(image_urls=firestore_property.media.image_urls)
                )
                save_property_task = firestore_service.save_property(firestore_property)
                
                # Execute image analysis and property save in parallel
                image_analysis, _ = await asyncio.gather(analysis_task, save_property_task)
                
                if image_analysis:
                    # Update analysis and get the final property in parallel
                    update_task = firestore_service.update_property_analysis(
                        firestore_property.listing_id, 
                        image_analysis
                    )
                    get_property_task = firestore_service.get_property(firestore_property.listing_id)
                    
                    # Wait for both operations to complete
                    _, analyzed_property = await asyncio.gather(update_task, get_property_task)
                    
                    if analyzed_property:
                        return analyzed_property
            except Exception as e:
                print(f"Error analyzing images for {firestore_property.listing_id}: {str(e)}")
            return None
        tasks.append(analyze_images())
    
    try:
        # Run all tasks concurrently and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any of the results is a valid property (from image analysis)
        for result in results:
            if isinstance(result, FirestoreProperty):
                return result
                
    except Exception as e:
        print(f"Error in concurrent processing for {firestore_property.listing_id}: {str(e)}")
    
    # If we get here, either there were no tasks or no valid property was returned
    # Save the property if it wasn't saved during image analysis
    if not stored_property:
        await firestore_service.save_property(firestore_property)
    
    return firestore_property

@tool
async def search_properties(search_params: dict) -> List[FirestoreProperty]:
    """Search for properties based on given criteria.
    
    Args:
        search_params: Dictionary containing search parameters that will be converted to PropertySearchRequest
            - location (List[str]): List of locations to search, format: STATE-SUBURB-POSTCODE
            - min_price (Optional[float]): Minimum price
            - max_price (Optional[float]): Maximum price
            - min_bedrooms (Optional[int]): Minimum number of bedrooms
            - min_bathrooms (Optional[int]): Minimum number of bathrooms
            - property_type (Optional[List[str]]): List of property types
            - car_parks (Optional[int]): Number of car parks
            - land_size_from (Optional[float]): Minimum land size in sqm
            - land_size_to (Optional[float]): Maximum land size in sqm
    
    Returns:
        List of FirestoreProperty objects containing search results and analysis
    """
    search_request = PropertySearchRequest(**search_params)
    results = await property_scraper.search_properties(search_request, max_results = 5)
    
    # Process all properties concurrently
    try:
        firestore_properties = await asyncio.gather(
            *[process_property(result) for result in results]
        )
        return [prop for prop in firestore_properties if prop is not None]
    except Exception as e:
        print(f"Error in concurrent property processing: {str(e)}")
        return []

@tool
async def get_property_recommendations(recommendation_params: Dict[str, Any]) -> PropertyRecommendationResponse:
    """Get personalized property recommendations based on analyzed properties and user preferences.
    
    Note: This tool expects the properties to be available in the state. The properties parameter
    in recommendation_params is ignored as we use the properties stored in state.
    
    Args:
        recommendation_params: Dictionary containing:
            - preferences (UserPreferences): User preferences for recommendation
            - recommendation_history (List[str]): List of previously recommended property listing_ids
    
    Returns:
        PropertyRecommendationResponse containing recommended properties
    """
    # This is just a placeholder - actual properties will be injected in tool_node
    return PropertyRecommendationResponse(properties=[])

@tool
async def process_preferences(session_id: str, user_message: str, context_type: str = "normal") -> dict:
    """Process user preferences and search parameters from conversation
    
    Args:
        session_id: str - The ID of the chat session
        user_message: str - The user's message to process
        context_type: str - Type of context ("normal" or "rejection"). If "rejection", the message is treated as property rejection feedback
        
    Returns:
        dict: A dictionary containing the updated preferences, search parameters, and any ambiguity information
    """
    service = PreferenceService()
    
    try:
        # Process user input and update preferences/search parameters
        preferences, search_params, ambiguity = await service.process_user_input(session_id, user_message)
        
        return {
            "preferences": preferences,
            "search_params": search_params,
            "ambiguity": ambiguity,
            "context_type": context_type
        }
    except Exception as e:
        print(f"Error processing preferences: {e}")
        return {
            "preferences": {},
            "search_params": {},
            "ambiguity": None,
            "context_type": context_type,
            "error": str(e)
        }

@tool
async def query_database(question: str, context: Optional[str] = None) -> dict:
    """Query the suburb database using natural language
    
    Args:
        question: str - The natural language question to query the database
        context: Optional[str] - Additional context to help generate the SQL query
    
    Returns:
        dict: A dictionary containing the query, results and any error messages
    """
    response = await sql_service.process_question(question, context)
    return response.model_dump()

search_tool = DuckDuckGoSearchResults()

# Augment the LLM with tools
tools = [
    search_properties,
    get_property_recommendations,
    process_preferences,
    search_tool,
    query_database
]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

async def get_conversation_context(session_id: str) -> str:
    """Generate context summary from conversation history in Firestore"""
    session = await chat_storage.get_session(session_id)
    if not session or not session.messages:
        return ""
    
    context = "\nPrevious conversation context:\n"
    # Get last 5 messages
    recent_messages = session.messages[-5:]
    for msg in recent_messages:
        context += f"- {msg.role}: {msg.content}\n"
    return context

async def llm_call(state: dict) -> AgentMessagesState:
    """LLM with conversation awareness"""
    session_id = state["session_id"]
    
    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    context = await get_conversation_context(session_id)
    
    response = llm_with_tools.invoke(
        [
            SystemMessage(
content=f"""
You are an intelligent property agent assistant that helps users find and analyze properties, answer questions about property market, suburbs, and related information.

Your job is to IMMEDIATELY use provided tools to fulfill user requests. 

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Important Tool Usage Guidelines:
1. Always use `process_preferences` first when user expresses preferences(Style, Environment, Features, Quality, Layout, Transport, Location, Investment) or requirements(location, price, property type, etc):
   - This tool will detect any ambiguous information that needs clarification
   - If ambiguity is detected, STOP and engage with the user to clarify
   - Present the suggestions and ask the follow-up question provided in the ambiguity info
   - Only proceed with property search after ambiguity is resolved

2. When handling ambiguous information:
   - Acknowledge the ambiguity clearly
   - Present the suggestions in a natural, conversational way
   - Explain why each suggestion might be suitable
   - Ask the follow-up question to get clarification
   - Wait for user's response before proceeding

3. Use `query_database` when user asks about:
   - Recommend a suburb based on user's requirments
   - Rental yield or rent income in a suburb
   - Median house price in a certain area
   - Property investment potential or historical growth
   - Suburb comparisons based on income, unemployment, or affluence
   - Family-friendliness, education, or demographics of a suburb
   - Distance to city or recommendations based on distance
   - Market supply (stock on market) or days on market

4. Use `search_tool` when you need additional information from web search
5. Use `search_properties` only after preferences and search parameters are clear, and location is explicitly mentioned
6. After `search_properties`, use `get_property_recommendations` for personalized recommendations

When NOT using tools, follow these guidelines for natural language responses:
1. Provide comprehensive analysis that goes beyond surface-level observations
2. Consider multiple perspectives and potential implications
3. Share relevant insights about market trends, demographic patterns, and future growth potential
4. Use clear, engaging language that builds rapport with the user
5. Structure responses to flow naturally from broad context to specific details
6. Include both quantitative data and qualitative insights when available
7. Acknowledge uncertainties and areas where more information might be needed
8. Offer thoughtful suggestions while respecting the collaborative nature of the conversation
9. Connect individual property features to broader lifestyle and investment considerations
10. Maintain a professional yet approachable tone that builds trust

DO NOT reply in plain text about what you "plan" or "will" do.

{context}
"""
            )
        ]
        + state["messages"]
    )
    
    return AgentMessagesState(
        messages=state["messages"] + [response],
        session_id=session_id,
        preferences=state["preferences"],
        search_params=state["search_params"]
    )

async def tool_node(state: Dict[str, Any]) -> AgentMessagesState:
    """Enhanced tool execution with history awareness"""
    session_id = state["session_id"]
    
    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    result: List[ToolMessage] = []
    recommendation_history: List[str] = state.get("recommendation_history", [])
    latest_recommendation: Optional[PropertyRecommendationResponse] = state.get("latest_recommendation")
    available_properties: List[FirestoreProperty] = state.get("available_properties", [])
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        
        # Add session_id and other state info to tool args
        if isinstance(args, dict):
            args["session_id"] = session_id
            if tool.name == "search_properties":
                args["search_params"] = state["search_params"]
        
        # Execute tool
        observation = await tool.ainvoke(args)
        
        # Special handling for preference processing with ambiguity
        if tool.name == "process_preferences" and isinstance(observation, dict):
            if observation.get("ambiguity"):
                # If ambiguity is detected, don't update state yet
                result.append(ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                ))
                # Return early to wait for user clarification
                return AgentMessagesState(
                    messages=state["messages"] + result,
                    session_id=session_id,
                    preferences=state["preferences"],
                    search_params=state["search_params"],
                    recommendation_history=recommendation_history,
                    latest_recommendation=latest_recommendation,
                    available_properties=available_properties
                )
            else:
                # Update state with new preferences and search parameters
                state["preferences"] = observation.get("preferences", state["preferences"])
                state["search_params"] = observation.get("search_params", state["search_params"])
        
        # Special handling for property recommendations
        elif tool.name == "get_property_recommendations":
            # Filter out previously recommended properties
            new_properties = [
                prop for prop in available_properties 
                if prop.listing_id not in recommendation_history
            ]
            
            if not new_properties:
                observation = PropertyRecommendationResponse(properties=[])
            else:
                # Get recommendations using the recommender service
                observation = await recommender.get_recommendations(
                    properties=new_properties,
                    preferences=state["preferences"]
                )
            
            # Update latest recommendation and history
            latest_recommendation = observation
            if isinstance(observation, dict):
                new_recommendations = [
                    prop.listing_id for prop in observation.get("properties", [])
                ]
                recommendation_history.extend(new_recommendations)
        
        elif tool.name == "search_properties":
            # Store the search results in state
            available_properties = available_properties + observation
        
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    
    return AgentMessagesState(
        messages=state["messages"] + result,
        session_id=session_id,
        preferences=state["preferences"],
        search_params=state["search_params"],
        recommendation_history=recommendation_history,
        latest_recommendation=latest_recommendation,
        available_properties=available_properties
    )

# Conditional edge function
def should_continue(state: dict) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END

# Build workflow
agent_builder = StateGraph(AgentMessagesState)  # Keep using MessagesState as base

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


