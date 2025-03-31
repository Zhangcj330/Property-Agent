from typing_extensions import Literal
from typing import List, Dict, Optional, Any
from datetime import datetime

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END

from app.config import settings
from app.models import PropertySearchRequest, PropertySearchResponse, UserPreferences, FirestoreProperty, ChatSession, ChatMessage
from app.services.image_processor import ImageProcessor, ImageAnalysisRequest, PropertyAnalysis
from app.services.property_scraper import PropertyScraper
from app.services.recommender import PropertyRecommender
from app.services.chat_storage import ChatStorageService
from IPython.display import Image, display

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
# Initialize services
image_processor = ImageProcessor()
property_scraper = PropertyScraper()
recommender = PropertyRecommender()
chat_storage = ChatStorageService()

# Default LLM
llm = ChatGoogleGenerativeAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash",
)

# Enhanced tools with history awareness
@tool
async def get_session_state(session_id: str) -> Dict[str, Any]:
    """Fetch session state including search parameters and preferences from Firestore.
    
    Args:
        session_id: Unique identifier for the conversation
    
    Returns:
        Dict containing search_params and preferences if they exist
    """
    session: Optional[ChatSession] = await chat_storage.get_session(session_id)
    if not session:
        return {"search_params": None, "preferences": None}
    
    return {
        "search_params": getattr(session, "search_params", None),
        "preferences": getattr(session, "preferences", None)
    }

@tool
async def search_properties(search_params: dict) -> List[PropertySearchResponse]:
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
        List of PropertySearchResponse objects containing search results
    """
    search_request = PropertySearchRequest(**search_params)
    return await property_scraper.search_properties(search_request)

@tool("analyze_property_images_tool", args_schema=ImageAnalysisRequest, return_direct=True)
async def analyze_property_images(
    request:ImageAnalysisRequest
) -> PropertyAnalysis:
    """Analyze property images to extract features and quality assessment.
    
    Args:
        ImageAnalysisRequest: List of URLs pointing to property images
    """
    image_processor = ImageProcessor()
    return await image_processor.analyze_property_image(request)

@tool
async def get_property_recommendations(
    session_id: str,
    properties: List[FirestoreProperty],
    preferences: Dict[str, any],
    limit: int = 5
) -> List[FirestoreProperty]:
    """Get personalized recommendations considering conversation history.
    
    Args:
        session_id: Unique identifier for the conversation
        properties: List of properties to analyze
        preferences: User preferences for property matching
        limit: Maximum number of recommendations to return
    """
    # Get stored preferences from Firestore
    session_state = await get_session_state(session_id)
    stored_prefs = session_state.get("preferences", {}) or {}
    
    # Merge current preferences with stored preferences
    enhanced_prefs = {**stored_prefs, **preferences}
    
    user_prefs = UserPreferences(**enhanced_prefs)
    recommendations = await recommender.get_recommendations(properties, user_prefs, limit)
    
    # Update session state with new preferences
    await chat_storage.update_session_state(
        session_id,
        preferences=enhanced_prefs
    )
    
    return recommendations

# Augment the LLM with tools
tools = [search_properties, analyze_property_images, get_property_recommendations, get_session_state]
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

# Enhanced nodes
async def llm_call(state: MessagesState):
    """LLM with conversation awareness"""
    session_id = state.get("session_id", "default")
    
    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    context = await get_conversation_context(session_id)
    
    response = llm_with_tools.invoke(
        [
            SystemMessage(
                content=f"""You are an intelligent property agent assistant that helps users find and analyze properties. 
                Your goal is to help users find suitable properties through a systematic workflow:

                1. Initial Property Search:
                   - Parse user input into structured PropertySearchRequest:
                   - If location missing or ambiguous, ask user for clarification
                   - Validate and normalize all numeric values
                   - Use search_properties tool with validated PropertySearchRequest
                
                2. Process Search Results:
                   - If no results, suggest broadening search criteria or alternative locations
                   - For results with images, use analyze_property_images tool to extract:
                     * Architectural style and era
                     * Build quality and condition
                     * Notable features and amenities
                     * Environmental factors
                
                3. Generate Recommendations:
                   - Combine search results with image analysis
                   - Use get_property_recommendations tool to:
                     * Match properties against user preferences
                     * Score and rank suitable properties
                     * Identify key highlights and potential concerns
                   - Present top recommendations with detailed explanations
                
                4. Interactive Refinement:
                   - If user needs are not met, adjust search criteria
                   - Consider user's flexibility on:
                     * Price range
                     * Location alternatives
                     * Property type
                     * Required features
                   
                Always maintain context between steps and use previous findings to inform next actions.
                Provide clear explanations for your recommendations and decisions.
                
                {context}
                """
            )
        ]
        + state["messages"]
    )
    
    # Log AI response and any tool results
    await chat_storage.save_message(
        session_id,
        ChatMessage(
            role="assistant",
            content=str(response.content),
            timestamp=datetime.now()
        )
    )
    
    return {"messages": [response]}

async def tool_node(state: dict):
    """Enhanced tool execution with history awareness"""
    session_id = state.get("session_id", "default")
    
    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    result = []
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        
        # Only add session_id for tools that need it
        if tool.name in ["get_property_recommendations", "get_session_state"]:
            if isinstance(args, dict):
                args["session_id"] = session_id
        
        observation = await tool.ainvoke(args)
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
        # Log tool execution in chat history
        await chat_storage.save_message(
            session_id,
            ChatMessage(
                role="system",
                content=f"Tool {tool.name} executed with result: {str(observation)[:100]}...",
                timestamp=datetime.now()
            )
        )
    
    return {"messages": result}

# Conditional edge function
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END

# Build workflow
agent_builder = StateGraph(MessagesState)

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
    
    # Create session if it doesn't exist
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    from app.models import UserPreference
    search_request = PropertySearchRequest(
        location=["NSW-chatswood-2067"],
        max_price=3500000,
        property_type=["house"],
        min_bedrooms=3
    )
    preferences = UserPreferences(
        Style = UserPreference(preference="modern", confidence_score=0.9, weight=0.5),
        Environment = UserPreference(preference="quiet", confidence_score=0.9, weight=0.5),
        Quality= UserPreference(preference="high", confidence_score=0.9, weight=0.5),
    )
    print("🏠 Starting Property Agent Example...")
    
    # Initial property search query
    initial_query = f"""
    recommend properties based in Sydney Chatswood 2067, with 3 bedrooms, max price 3.5M, and modern style, quiet environment, and high quality.
    """
    
    messages = [HumanMessage(content=initial_query)]
    result = await agent.ainvoke({
        "messages": messages,
        "session_id": session_id
    })
    
    print("\n🤖 Agent's Response:")
    for message in result["messages"]:
        message.pretty_print()

# Run the example if this file is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_example())

