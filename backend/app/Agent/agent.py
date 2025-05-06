import os
import sys
from pathlib import Path
import asyncio
import json
import logging

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

from typing_extensions import Literal
from typing import List, Dict, Optional, Any, Annotated
from datetime import datetime

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
from app.services.preference_service import PreferenceService
from app.services.planning_service import get_planning_info
from app.services.investment_service import InvestmentService
from app.services.duckduckgo_search import DuckDuckGoSearchResults
from app.services.sql_service import SQLService
from IPython.display import Image, display
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

response_llm = ChatGoogleGenerativeAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash",
)

logger = logging.getLogger(__name__)

# Define custom state class
class AgentMessagesState(MessagesState):
    """State class for the agent that extends MessagesState with additional fields"""
    session_id: str
    preferences: Dict[str, Any]
    search_params: Dict[str, Any]
    available_properties: List[FirestoreProperty] = Field(default_factory=list, description="List of properties from the latest search")
    latest_recommendation: Optional[PropertyRecommendationResponse] = Field(default=None, description="Last recommendation")

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
                logger.error(f"Error getting planning info for {firestore_property.listing_id}: {str(e)}")
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
                logger.error(f"Error getting investment metrics for {firestore_property.listing_id}: {str(e)}")
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
                logger.error(f"Error analyzing images for {firestore_property.listing_id}: {str(e)}")
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
        logger.error(f"Error in concurrent processing for {firestore_property.listing_id}: {str(e)}")
    
    # If we get here, either there were no tasks or no valid property was returned
    # Save the property if it wasn't saved during image analysis
    if not stored_property:
        await firestore_service.save_property(firestore_property)
    
    return firestore_property

@tool
async def search_properties(
    location: Annotated[List[str], Field(description="List of locations to search, format: [STATE-SUBURB-POSTCODE, STATE-SUBURB-POSTCODE, ...]")],
    min_price: Annotated[Optional[float], Field(description="Minimum price")] = None,
    max_price: Annotated[Optional[float], Field(description="Maximum price")] = None,
    min_bedrooms: Annotated[Optional[int], Field(description="Minimum number of bedrooms")] = None,
    min_bathrooms: Annotated[Optional[int], Field(description="Minimum number of bathrooms")] = None,
    property_type: Annotated[Optional[List[str]], Field(description="List of property types (house, apartment, unit, townhouse, villa, rural)")] = None,
    car_parks: Annotated[Optional[int], Field(description="Number of car parks")] = None,
    land_size_from: Annotated[Optional[float], Field(description="Minimum land size in sqm")] = None,
    land_size_to: Annotated[Optional[float], Field(description="Maximum land size in sqm")] = None
) -> List[FirestoreProperty]:
    """Search on listing website for properties based on given search parameters.
    
    Args:
        location (List[str]): List of locations to search, format: [STATE-SUBURB-POSTCODE, STATE-SUBURB-POSTCODE, ...]
        min_price (Optional[float]): Minimum price
        max_price (Optional[float]): Maximum price
        min_bedrooms (Optional[int]): Minimum number of bedrooms
        min_bathrooms (Optional[int]): Minimum number of bathrooms
        property_type (Optional[List[str]]): List of property types (house, apartment, unit, townhouse, villa, rural)
        car_parks (Optional[int]): Number of car parks
        land_size_from (Optional[float]): Minimum land size in sqm
        land_size_to (Optional[float]): Maximum land size in sqm
    
    Returns:
        List of FirestoreProperty objects containing listing properties from the search results. 
    """
    search_params = {
        "location": location,
        "min_price": min_price,
        "max_price": max_price,
        "min_bedrooms": min_bedrooms,
        "min_bathrooms": min_bathrooms,
        "property_type": property_type,
        "car_parks": car_parks,
        "land_size_from": land_size_from,
        "land_size_to": land_size_to
    }
    search_request = PropertySearchRequest(**search_params)
    results = await property_scraper.search_properties(search_request, max_results=5)
    try:
        firestore_properties = await asyncio.gather(
            *[process_property(result) for result in results]
        )
        return [prop for prop in firestore_properties if prop is not None]
    except Exception as e:
        logger.error(f"Error in concurrent property processing: {str(e)}")
        return []

@tool
async def recommend_from_available_properties() -> PropertyRecommendationResponse:
    """Recommend the best properties from the current available_properties list in state, based on user preferences.
    
    Precondition: This tool should only be used if the state['available_properties'] is not empty. It will generate recommendations from the properties that have just been searched and are available in the current session state.
    
    Returns:
        PropertyRecommendationResponse containing recommended properties
    """
    # This is just a placeholder - actual properties will be injected in tool_node
    return PropertyRecommendationResponse(properties=[])

@tool
async def process_preferences(session_id: str, user_message: str) -> dict:
    """Update user preferences and search parameters based on user's conversation
    
    Args:
        session_id: str - The ID of the chat session
        user_message: str - The user's message to process
        
    Returns:
        dict: A dictionary containing the updated preferences, search parameters
    """
    service = PreferenceService()
    
    try:
        # Process user input and update preferences/search parameters
        preferences, search_params = await service.process_user_input(session_id, user_message)
        
        return {
            "preferences": preferences,
            "search_params": search_params
        }
    except Exception as e:
        logger.error(f"Error processing preferences: {e}")
        return {
            "preferences": {},
            "search_params": {},
            "error": str(e)
        }
@tool
async def search_suburb(
    question: str,
    filters: Optional[Dict[str, Any]] = None
) -> dict:
    """Use this tool when the user expresses general investment or home-buying preferences without specifying a particular suburb. 
    This tool helps explore potential suburbs based on their stated needs and desires (e.g., family-friendly, high growth potential, good rental yield) at a broader city or regional level, rather than focusing on specific streets or neighborhoods.
    
    Args:
        question: str - The natural language question to query the database
        filters: Optional[Dict[str, Any]] = None - Dictionary of filter conditions for suburb search
                Supported filters:
                - state: str - State abbreviation (e.g., "NSW", "VIC")
                - min_price/max_price: float - Price range for properties
                - min_rental_yield/max_rental_yield: float - Rental yield percentage
                - min_growth/max_growth: float - Property value growth rate
                - distance_to_cbd: float - Maximum distance to CBD in km
                - min_income/max_income: float - Weekly household income range
                - family_percentage: float - Percentage of family households
                - vacancy_rate: float - Property vacancy rate
                - days_on_market: int - Average days properties stay on market
    
    Returns:
        dict: A dictionary containing the query, results and any error messages
    """
    response = await sql_service.process_question(question, filters=filters)
    return response.model_dump()

web_search = DuckDuckGoSearchResults()

# Augment the LLM with tools
tools = [
    search_properties,
    recommend_from_available_properties,
    process_preferences,
    web_search,
    search_suburb
]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

async def get_conversation_context(session_id: str) -> str:
    """Generate context summary from conversation history in Firestore"""
    session = await chat_storage.get_session(session_id)
    if not session or not session.messages:
        return ""
    
    context = "\nPrevious conversation context:\n"
    # Get last 10 messages but only include user and assistant messages
    recent_messages = session.messages[-10:]
    for msg in recent_messages:
        if msg.role in ["assistant", "user"]:
            context += f"- {msg.role}: {msg.content}\n"
    return context

async def llm_call(state: dict) -> AgentMessagesState:
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
Begin by asking the user to clarify their motivation: are they seeking an investment property or purchasing a home to live in?

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Important Tool Usage Guidelines:
1. *Must* use `process_preferences` in the following situations:
   - When the user expresses preferences (location, price, property type, Style, Environment, Features, Quality, Layout, Transport, Location, Investment).
   - When the user accepts suggestions based on the latest conversation.
   - When the new conversation content contains additional or updated preference information.

2. Use `search_suburb` when recommend a suburb based on user's preferences
   Please Heuristically and conversationally guide the user in an open-ended way to describe their property investment preferences, 
   including budget, preferred location (states, cities), investment potential, historical growth, rental yield, family-friendliness, average income, 
   unemployment, demographics, affluence, and days on market, distance to city center. 
   Let the user answer freely. Extract as many preferences as possible from the user's natural language response.

3. Use `web_search` when you need additional information from web
    - Economic strength
    - Infrastructure pipeline
    - Industry mix (health, education, etc.)
    - Market tightness (inventory/vacancy context)
    - Relevant insights about market trends, demographic patterns, and future growth potential

4. Use `search_properties` after preferences and search parameters are clear, especially location make sure format is correct.
    - never ask user to provide postcode, search web for postcode if you need it.
    - before using this tool, ask the user to clarify all perference and search parameters. Confirm whether they'd like to add or adjust any criteria in their property search
    - proactively guide user to find suburbs use `search_suburb` and `web_search` when user does not have a strong location preference

5. After `search_properties`, *Must* use `recommend_from_available_properties` for personalized recommendations

Find out as soon as possible whether the user has a preferred area. 
If there are any, recommend the houses in this area. 
If the user is not clear about the area, ask about the preferences related to the area as soon as possible and use the search_suburb and web_search tools for recommendations

When NOT using tools, follow these guidelines for natural language responses:
1. Provide comprehensive analysis that goes beyond surface-level observations
2. Consider multiple perspectives and potential implications
4. Use clear, engaging language that builds rapport with the user
5. Structure responses to flow naturally from broad context to specific details
6. Include both quantitative data and qualitative insights when available
7. Acknowledge uncertainties and areas where more information might be needed
8. Offer thoughtful suggestions while respecting the collaborative nature of the conversation
9. Connect individual property features to broader lifestyle and investment considerations
10. Maintain a professional yet approachable tone that builds trust
11. Proactively guide users to clarify their needs.
12. Proactively ask user if they need help on provide suburb information. if so, utilize `search_suburb` and `web_search` to recommend suburb.
13. When making a recommendation, always provide: Specific data or evidence and sources or references for the evidence provided.
14. If ambiguity is detected, STOP and engage with the user to clarify
15. only ask one question at a time, Provide context or examples if helpful.

DO NOT reply in plain text about what you "plan" or "will" do. 
Do not mention the tool you are using in your response.


previous user's preferences: 
{state["preferences"]}

previous user's requirements:
{state["search_params"]}

available properties:
{state["available_properties"]}

conversation context:
{context}
"""
            )
        ]
        + state["messages"] # single run message, tools call from latest user message
    )
    
    return AgentMessagesState(
        messages=state["messages"] + [response],
        session_id=session_id,
        preferences=state["preferences"],
        search_params=state["search_params"],
        latest_recommendation=state["latest_recommendation"]
    )

async def tool_node(state: Dict[str, Any]) -> AgentMessagesState:
    session_id = state["session_id"]
    # 初始化所有 return 需要的变量，避免 UnboundLocalError
    preferences = state.get("preferences", {})
    search_params = state.get("search_params", {})
    latest_recommendation = state.get("latest_recommendation", None)
    available_properties: List[FirestoreProperty] = state.get("available_properties", [])

    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    result: List[ToolMessage] = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        
        # Add tool usage logging - Replace print with logger
        logger.info(f"Using tool: {tool.name}")
        logger.info(f"Tool arguments: {json.dumps(args, indent=2)}")
        
        # Add session_id and other state info to tool args
        if isinstance(args, dict):
            args["session_id"] = session_id
        
        # Execute tool
        observation = await tool.ainvoke(args)
        
        # Special handling for preference processing with ambiguity
        if tool.name == "process_preferences" and isinstance(observation, dict):
            # Update state with new preferences and search parameters
            preferences = observation.get("preferences", preferences)
            search_params = observation.get("search_params", search_params)

        # Special handling for property recommendations
        elif tool.name == "recommend_from_available_properties":
            # Get previously recommended properties from chat storage
            recommendation_history = await chat_storage.get_recommendation_history(session_id)
            # Filter out previously recommended properties
            new_properties = [
                prop for prop in available_properties 
                if prop.listing_id not in recommendation_history
            ]
            
            if not new_properties:
                observation = "No new properties to recommend. need to search properties again."
            else:
                # Get recommendations using the recommender service
                observation = await recommender.get_recommendations(
                    properties=new_properties,
                    preferences=preferences
                )
                latest_recommendation = observation
                # Replace print with logger
                logger.info(f"Last recommendation: {latest_recommendation}")
            # Save as ChatMessage
            chat_msg = ChatMessage(
                role="tool",
                content="Property recommendations generated.",  # 简要摘要
                type="property_recommendation",
                recommendation=latest_recommendation if isinstance(latest_recommendation, PropertyRecommendationResponse) else None,
                metadata={"tool_call_id": tool_call["id"], "tool_name": tool.name},
                timestamp=datetime.now()
            )
            await chat_storage.save_message(session_id, chat_msg)

        elif tool.name == "search_properties":
            available_properties = observation
            # Update available_properties state in chat storage
            await chat_storage.update_recommendation_state(
                session_id=session_id,
                available_properties=available_properties
            )

        result.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            metadata={"type": "property_recommendation"}
        ))

    return AgentMessagesState(
        messages=state["messages"] + result,
        session_id=session_id,
        preferences=preferences,
        search_params=search_params,
        available_properties=available_properties,
        latest_recommendation=latest_recommendation
    )

# Conditional edge function
def should_continue(state: dict) -> Literal["environment", "response"]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return "response"

async def response_node(state: dict) -> AgentMessagesState:
    """Generate natural language response based on the current state.
    This node is responsible for:
    1. Formatting the final response to the user
    2. Saving the response to chat storage
    3. Applying any response-specific processing or formatting
    """
    session_id = state["session_id"]

    context = await get_conversation_context(session_id)
    response = response_llm.invoke([
        SystemMessage(content=f"""
As a property agent assistant, follow these guidelines for natural language responses:
1. Make responses natural and engaging while maintaining a professional tone
2. Ask one question at a time
3. Offer thoughtful suggestions while respecting the collaborative nature of the conversation
                  
Find out as soon as possible whether the user has a preferred area. 
If there are any, recommend the houses in this area. 
If the user is not clear about the area, ask about the preferences related to the area as soon as possible and use the search_suburb and web_search tools for recommendations

You should ask the user a structured series of questions to fully understand their intent and preferences.

Branch based on the answer:
If the user says **investment**:
Ask these questions step by step:
1. What is your primary investment goal? (e.g., capital growth, rental yield, or both)
2. What is your maximum budget?
3. Do you have a minimum expected rental yield? (e.g., 4.5% or higher)
4. Are there specific regions or towns you're interested in? (e.g., any particular state, LGA, suburb)
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
5. What is the maximum distance from the town center you're comfortable with? (e.g., 15 km)
6. Which local economic features are important to you? (e.g., infrastructure development, population growth, job diversity in sectors like health or education)
7. Are there any areas or types of locations you'd like to avoid? (e.g., tourist towns, affluent suburbs, remote regions)
after asking all the questions, use `search_properties` to get the properties.

If the user says **home buyer**:
As an outstanding real estate agent helping buyers select their own homes, you should guide users to focus on the following core directions
Make sure to understand both the basic conditions of the buyers and uncover their potential true needs
1. Motivation and Timeline
2. Budget and Financial Framework
3. Location Preferences
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
4. Schools and Family Planning
5. Property Type and Functional Needs
6. Lifestyle and Environmental Preferences
7. Property Condition and Style
8. Must-Haves and Deal Breakers
                      
conversation context:
{context}
"""),
        HumanMessage(content=f"Please revise this message imitate the property buyer's agent: {state['messages'][-1].content}")
    ])
    
    return AgentMessagesState(
        messages=state["messages"] + [response],
        session_id=session_id,
        preferences=state.get("preferences", {}),
        search_params=state.get("search_params", {}),
        available_properties=state.get("available_properties", []),
        latest_recommendation=state.get("latest_recommendation", None)
    )

# Build workflow
agent_builder = StateGraph(AgentMessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)
agent_builder.add_node("response", response_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        "response": "response",
    },
)
agent_builder.add_edge("environment", "llm_call")
agent_builder.add_edge("response", END)

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

"""
You must ask the user a structured series of questions to fully understand their intent and preferences.
Begin by asking the user to clarify their motivation: are they seeking an investment property or purchasing a home to live in?

Branch based on the answer:
If the user says **investment**:
Ask these questions step by step:
1. What is your primary investment goal? (e.g., capital growth, rental yield, or both)
2. What is your maximum budget?
3. Do you have a minimum expected rental yield? (e.g., 4.5% or higher)
4. Are there specific regions or towns you're interested in? (e.g., any particular state, LGA, suburb)
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
5. What is the maximum distance from the town center you're comfortable with? (e.g., 15 km)
6. Which local economic features are important to you? (e.g., infrastructure development, population growth, job diversity in sectors like health or education)
7. Are there any areas or types of locations you'd like to avoid? (e.g., tourist towns, affluent suburbs, remote regions)
after asking all the questions, use `search_properties` to get the properties.

If the user says **home buyer**:
As an outstanding real estate agent helping buyers select their own homes, you should guide users to focus on the following core directions
Make sure to understand both the basic conditions of the buyers and uncover their potential true needs
1. Motivation and Timeline
2. Budget and Financial Framework
3. Location Preferences
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
4. Schools and Family Planning
5. Property Type and Functional Needs
6. Lifestyle and Environmental Preferences
7. Property Condition and Style
8. Must-Haves and Deal Breakers

"""