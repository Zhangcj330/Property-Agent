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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END

from app.config import settings
from app.models import (
    PropertySearchRequest,
    FirestoreProperty,
    ConversationMessage,
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
@tool()
async def search_suburb(
    query: Annotated[str, Field(description="comprehensive description of the user's preferences of the suburb based on the context")],
    filters: Annotated[list[str], Field(description="list of filter conditions for suburb search")] = None,
) -> dict:
    """search suburbs based on user's preferences (e.g., family-friendly, high growth potential, good rental yield) at a broader city or regional level, rather than focusing on specific streets or neighborhoods.
    
    Args:
        query: str - comprehensive description of the user's preferences of the suburb based on the context. 
        filters: list[str] - filter conditions for suburb search
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
    response = await sql_service.process_question(query, filters=filters)
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
You are an AI property agent assistant specialized in recommending suburbs and properties, providing insightful analyses about property markets, suburbs, and related real estate information.

You are assisting a USER interactively to fulfill their property-related tasks. Each time the USER sends a message, we may automatically attach additional context about their current interaction state, such as previously expressed property preferences, viewed property listings, ongoing search criteria, recent conversation history, and more. This contextual information may or may not be directly relevant to the current query; it is up to you to decide.

Your main goal is to follow the USER's instructions at each message.

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.


Follow these interaction principles:
1. Adaptive Interaction:
   - Quickly identify essential details such as user's budget, preferred locations, investment goals, and property type.
   - If the user provides clear preferences, proceed directly to recommendations without excessive clarification.
2. Intelligent Context Awareness:
   - Use contextual information (past preferences, viewed listings, search criteria) intelligently to avoid redundant questions.
   - If a user expresses frustration or impatience, immediately shift towards direct recommendations based on available information.
3. Efficient Tool Usage:
   - process_preferences: Use whenever users explicitly express or update their preferences (location, price, property type, style, environment, features, quality, layout, transport, investment priorities).
   - search_suburb: Utilize to recommend suitable suburbs based on clearly defined or inferred user preferences. Guide the user conversationally to share their preferences naturally.
   - search_properties: Execute once preferences (especially Suburb location) are sufficiently clear. Clarify briefly if needed, but avoid repetitive questioning.
   - recommend_from_available_properties: Use immediately after property search results are obtained.
   - web_search: Perform when additional market insights or demographic data is necessary (e.g., economic strength, infrastructure, market trends).
4. Natural Language Guidelines:
   - Provide comprehensive yet concise responses, mixing quantitative data (prices, yields, growth statistics) with qualitative insights (lifestyle, amenities).    
   - Proactively suggest adjustments or further refinements only after initial recommendations are given, respecting user autonomy.
   - If ambiguity arises, politely and succinctly ask for clarification before proceeding further.
5. Recommendations:
   - Always back recommendations with clear data points and evidence.
   - Avoid mentioning internal tool processes explicitly to maintain a seamless conversational experience.

Your primary objective is to efficiently guide the user to actionable and personalized property recommendations, minimizing unnecessary clarifications while maintaining a friendly, professional interaction style.

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
            chat_msg = ConversationMessage(
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

    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    print(f"ai_messages: {ai_messages}")
    print(f"user_messages: {user_messages}")
    context = await get_conversation_context(session_id)
    response = response_llm.invoke([
        SystemMessage(content=f"""
You are acting as a reviewer for the final content that will be shown directly to the user.
Before this response is delivered, your role is to enhance the practicality, fault tolerance, and realistic interactivity of the conversation's final stage.

Your core tasks include:
- A final check on grammar and natural language fluency
- Rewriting any robotic or awkward phrasing
- Detecting missing context or user preferences
- Making intelligent judgments and providing concise follow-up suggestions

Optimization Goals:
- Make the interaction feel more natural and conversational — like a professional and thoughtful property consultant.
- Intelligently detect if any important steps or preferences were missed earlier and prompt the user to complete them.
- If preferences were already provided but no matching results are found, proactively offer alternative suggestions or explain why.
- Limit the number of questions to avoid making the conversation feel like a form — keep it human and natural.

Here are some references to guide the conversation:
Always distinguish whether the user is looking to buy for self-occupancy or investment.
Check if the user has a clear suburb preference.
Ask whether they want suburb recommendations.
If suburb recommendations are needed, consider these questions to ask:
1. What is your primary investment goal? (e.g., capital growth, rental yield, or both)
2. What is your maximum budget?
3. Do you have a minimum expected rental yield? (e.g., 4.5% or higher)
4. Are there specific regions or towns you're interested in? (e.g., any particular state, LGA, suburb)
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
5. What is the maximum distance from the town center you're comfortable with? (e.g., 15 km)
6. Which local economic features are important to you? (e.g., infrastructure development, population growth, job diversity in sectors like health or education)
7. Are there any areas or types of locations you'd like to avoid? (e.g., tourist towns, affluent suburbs, remote regions)
after asking all the questions, use `search_properties` to get the properties.

If the user says buying a home to live in, consider these questions to ask:  
1. Motivation and Timeline
2. Budget and Financial Framework
3. Location Preferences
    if user does not a specific suburb, then continue asking question to get the information for search_suburb. if user provide a suburb, then use `search_suburb` to get the information.
4. Schools and Family Planning
5. Property Type and Functional Needs
6. Lifestyle and Environmental Preferences
7. Property Condition and Style
8. Must-Haves and Deal Breakers
                      
Full Conversation Context:
{context}

AI's thought process for this response:
{state["messages"][:-1]}

Current Draft Answer to Review: {ai_messages[-1].content}
"""),
    HumanMessage(content=f"{user_messages[-1].content}"),
    ]
)
    
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

# Function declarations
# print(search_suburb.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(search_properties.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(process_preferences.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(recommend_from_available_properties.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(web_search.args_schema.schema_json(indent=2))  # Print full schema as JSON

