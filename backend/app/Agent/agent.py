import os
import sys
from pathlib import Path

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
)
from app.services.image_processor import ImageProcessor, ImageAnalysisRequest, PropertyAnalysis
from app.services.property_scraper import PropertyScraper
from app.services.recommender import PropertyRecommender
from app.services.chat_storage import ChatStorageService
from app.services.firestore_service import FirestoreService
from app.services.preference_service import (
    extract_preferences_and_search_params,
    get_current_preferences_and_search_params,
    infer_preference_from_rejection,
    PreferenceService,
    PreferenceUpdate
)
from IPython.display import Image, display

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

# Initialize services
image_processor = ImageProcessor()
property_scraper = PropertyScraper()
recommender = PropertyRecommender()
chat_storage = ChatStorageService()
firestore_service = FirestoreService()

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
        List of FirestoreProperty objects containing search results with analysis
    """
    search_request = PropertySearchRequest(**search_params)
    results = await property_scraper.search_properties(search_request, max_results = 5)
    
    firestore_properties = []
    for result in results:
        # Convert to FirestoreProperty
        firestore_property = FirestoreProperty.from_search_response(result)
        
        # Check if property exists in Firestore with analysis
        stored_property = await firestore_service.get_property(firestore_property.listing_id)
        
        if stored_property and stored_property.analysis:
            # Use existing analysis if available
            firestore_properties.append(stored_property)
        elif firestore_property.media.image_urls:
            try:
                # Create new analysis
                image_analysis = await image_processor.analyze_property_image(
                    ImageAnalysisRequest(image_urls=firestore_property.media.image_urls)
                )
                # Save property and update with analysis
                await firestore_service.save_property(firestore_property)
                await firestore_service.update_property_analysis(
                    firestore_property.listing_id, 
                    image_analysis
                )
                # Get the updated property with analysis
                analyzed_property = await firestore_service.get_property(firestore_property.listing_id)
                if analyzed_property:
                    firestore_properties.append(analyzed_property)
            except Exception as e:
                print(f"Error analyzing property {firestore_property.listing_id}: {str(e)}")
                # Still include the property even if analysis fails
                firestore_properties.append(firestore_property)
        else:
            # Include property without analysis if no images available
            firestore_properties.append(firestore_property)
    
    return firestore_properties

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
async def extract_preferences(session_id: str, user_message: str) -> dict:
    """Extract preferences and search parameters from user message
    
    Args:
        session_id: str - The ID of the chat session
        user_message: str - The user's message containing preferences and search parameters
    
    Returns:
        dict: A dictionary containing the extracted preferences and search parameters
    """
    service = PreferenceService()
    
    # Get or create session
    session = await service.chat_storage.get_session(session_id)
    if not session:
        session = await service.chat_storage.create_session(session_id)
    
    # Save user message
    await service.chat_storage.save_message(
        session_id,
        ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now()
        )
    )
    
    # Extract preferences and search parameters
    preferences, search_params = await service.extract_from_context(
        session_id=session_id,
        recent_message=user_message
    )
    print(f"Extracted preferences: {preferences}")
    print(f"Extracted search parameters: {search_params}")
    
    # Update preferences and search parameters
    updated_preferences = None
    if preferences:
        updated_preferences = await service.update_user_preferences(session_id, preferences)
    
    updated_search_params = None
    if search_params:
        updated_search_params = await service.update_search_params(session_id, search_params)
    
    return {
        "session_id": session_id,
        "updated_preferences": updated_preferences,
        "updated_search_params": updated_search_params
    }

@tool
async def handle_rejection(session_id: str, rejection_message: str, property_details: Dict[str, Any]) -> dict:
    """Handle property rejection and infer preferences"""
    service = PreferenceService()
    
    # Get session and history
    session = await service.chat_storage.get_session(session_id)
    if not session:
        session = await service.chat_storage.create_session(session_id)
    
    # Prepare context
    context = service._prepare_conversation_context(session.messages)
    
    # Add rejected property information
    property_context = "Rejected Property Information:\n"
    for key, value in property_details.items():
        property_context += f"- {key}: {value}\n"
    
    # Use LLM to extract preferences from rejection
    response = service.llm.invoke([
        SystemMessage(content="""You are a professional property preference analyst. The user has just rejected a property recommendation. Please analyze potential reasons for rejection and infer implicit preferences. Focus on:
1. Negative words used ("don't like", "not suitable", etc.)
2. Property features potentially implied (location, style, environment, etc.)
3. Mismatches between user's previous preferences and this property

You must respond with ONLY a valid JSON array, without any additional text, explanation, or formatting. The array should follow this exact format:
[
  {
    "preference_type": "implicit",
    "category": "Style|Environment|Features|...",
    "value": "specific preference content",
    "importance": 0.5,
    "reason": "inference reason",
    "source_message": "rejection message"
  }
]

Example valid response:
[{"preference_type":"implicit","category":"Transport","value":"close to train station","importance":0.5,"reason":"User complained about distance to station","source_message":"too far from station"}]

If no preferences can be reliably inferred, respond with exactly: []"""),
        HumanMessage(content=f"Rejection Message: {rejection_message}\nProperty Information: {property_context}")
    ])
    
    try:
        # Parse response
        result = json.loads(response.content)
        
        # Apply these implicit preferences
        updates = [PreferenceUpdate(**item) for item in result]
        updated_preferences = {}
        
        if updates:
            updated_preferences = await service.update_user_preferences(session_id, updates)
        
        return {
            "preferences": [update.model_dump() for update in updates],
            "updated_preferences": updated_preferences
        }
    except Exception as e:
        print(f"Error inferring preferences from rejection: {e}")
        print(f"Raw response: {response.content}")
        return {"preferences": [], "updated_preferences": {}}

# Augment the LLM with tools
tools = [
    search_properties,
    get_property_recommendations,
    extract_preferences,
    handle_rejection
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
You are an intelligent property agent assistant that helps users find and analyze properties.

Your job is to IMMEDIATELY use provided tools to fulfill user requests. 

You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Important Tool Usage Guidelines:
- Always use `extract_preferences` first when user expresses preferences or requirements
- When extracted location is ambiguous (e.g., too large like "Sydney", "North Shore"), ask for clarification.
- Use `handle_rejection` when user expresses dissatisfaction with a property
- Always give recommendation about surburb based on user's preference, unless user specify otherwise.
- Use `search_properties` to find properties matching the search criteria
- After `search_properties`, use `get_property_recommendations` to get personalized property recommendations

DO NOT reply in plain text about what you "plan" or "will" do. Instead, IMMEDIATELY trigger the relevant tool action.

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
        
        # Special handling for property recommendations
        if tool.name == "get_property_recommendations":
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
        else:
            # Normal tool execution
            observation = await tool.ainvoke(args)
        
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
        # Update state based on tool results
        if tool.name == "extract_preferences" and isinstance(observation, dict):
            state["preferences"] = observation.get("updated_preferences", state["preferences"])
            state["search_params"] = observation.get("updated_search_params", state["search_params"])
        elif tool.name == "search_properties":
            # Store the search results in state
            available_properties = observation
        elif tool.name == "get_property_recommendations":
            # Update latest recommendation and history
            latest_recommendation = observation
            if isinstance(observation, dict):
                new_recommendations = [
                    prop.listing_id for prop in observation.get("properties", [])
                ]
                recommendation_history.extend(new_recommendations)
    
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

# Example usage
async def run_example():
    session_id = "example_session"

    # Example search request
    search_request = PropertySearchRequest(
        location=["NSW-chatswood-2067"],
        max_price=3500000,
        property_type=["house"],
        min_bedrooms=3
    )
    
    # Example user preferences
    preferences = UserPreferences(
        Style=UserPreference(preference="modern", confidence_score=0.9, weight=0.5),
        Environment=UserPreference(preference="quiet", confidence_score=0.9, weight=0.5),
        Quality=UserPreference(preference="high", confidence_score=0.9, weight=0.5),
        Features=UserPreference(preference="spacious", confidence_score=0.8, weight=0.4),
        Layout=UserPreference(preference="open plan", confidence_score=0.8, weight=0.4),
        Transport=UserPreference(preference="close to station", confidence_score=0.7, weight=0.3)
    )
    
    print("🏠 Starting Property Agent Example...")
    
    # Initial property search query with structured format
    initial_query = """
    Find properties in Chatswood (NSW-chatswood-2067) with these criteria:
    - Maximum price: $3.5M
    - Property type: house
    - Minimum 3 bedrooms
    
    Preferences:
    - Modern style (high priority)
    - Quiet environment (high priority)
    - High quality construction (high priority)
    - Spacious and open plan layout
    - Close to public transport
    
    Please analyze the properties and provide detailed recommendations.
    """
    
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "session_id": session_id,
        "preferences": preferences or {},
        "search_params": search_request or {},
        "recommendation_history": [],
        "latest_recommendation": None,
        "available_properties": []
    }
    final_state = await agent.ainvoke(initial_state)
    
    print("\n🤖 Agent's Response:")
    for message in final_state["messages"]:
        message.pretty_print()

async def test_preference_service():
    """Test function for preference service functionality"""
    try:
        session_id = "test_session"
        
        print("\n=== Testing Preference Service ===\n")
        
        # Test 1: Extract preferences from initial message
        print("Test 1: Extracting preferences from initial message")
        initial_message = """
        I'm looking for a house in Sydney, preferably modern style with 3 bedrooms.
        It should be in a quiet area and close to public transport.
        My budget is around 2.5M.
        """
        
        try:
            result = await extract_preferences.ainvoke({
                "session_id": session_id,
                "user_message": initial_message
            })
            
            print("\nExtracted Preferences:")
            if "preferences" in result:
                for pref in result["preferences"]:
                    print(f"• {pref['category']}: {pref['value']} (Importance: {pref['importance']})")
            
            print("\nSearch Parameters:")
            if "search_params" in result:
                for param in result["search_params"]:
                    print(f"• {param['param_name']}: {param['value']}")
            
            if "clarification" in result:
                print("\nLocation Clarification Needed:")
                location_info = result["clarification"]["location_info"]
                print(f"Term: {location_info['term']}")
                print("Suggestions:")
                for suggestion in location_info["suggestions"]:
                    print(f"• {suggestion}")
                print(f"Context: {location_info['context']}")
        except Exception as e:
            print(f"Error in Test 1: {str(e)}")
        
        # Test 2: Handle property rejection
        print("\nTest 2: Processing rejection feedback")
        rejection_message = "The house is too far from the train station and the area is quite noisy"
        property_details = {
            "listing_id": "test123",
            "address": "123 Test St, Sydney",
            "transport": "20 min walk to station",
            "environment": "Main road location"
        }
        
        try:
            rejection_result = await handle_rejection.ainvoke({
                "session_id": session_id,
                "rejection_message": rejection_message,
                "property_details": property_details
            })
            
            print("\nInferred Preferences from Rejection:")
            if "preferences" in rejection_result:
                for pref in rejection_result["preferences"]:
                    print(f"• {pref['category']}: {pref['value']}")
                    print(f"  Reason: {pref['reason']}")
        except Exception as e:
            print(f"Error in Test 2: {str(e)}")
        
        # Test 3: Get current state
        print("\nTest 3: Getting current preferences and search parameters")
        try:
            current_state = await get_session_state.ainvoke({
                "session_id": session_id
            })
            
            print("\nCurrent Preferences:")
            if current_state["preferences"]:
                for category, pref in current_state["preferences"].items():
                    print(f"• {category}: {pref.get('preference', 'Not set')}")
            
            print("\nCurrent Search Parameters:")
            if current_state["search_params"]:
                for param, value in current_state["search_params"].items():
                    print(f"• {param}: {value}")
        except Exception as e:
            print(f"Error in Test 3: {str(e)}")
            
    except Exception as e:
        print(f"Test execution error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
    
    async def run_tests():
        print("🏠 Starting Property Agent Tests...")
        await test_preference_service()
        print("\n✅ Tests completed!")
    
    # Run tests instead of example
    asyncio.run(run_tests())

