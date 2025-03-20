from typing_extensions import Literal
from typing import List, Dict, Optional

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END

from app.config import settings
from app.models import PropertySearchRequest, PropertySearchResponse, UserPreferences, FirestoreProperty
from app.services.image_processor import ImageProcessor, ImageAnalysisRequest
from app.services.property_scraper import PropertyScraper
from app.services.recommender import PropertyRecommender
from IPython.display import Image, display

# Initialize services
image_processor = ImageProcessor()
property_scraper = PropertyScraper()
recommender = PropertyRecommender()

# Default LLM
llm = ChatGoogleGenerativeAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash",
)

# Define tools
@tool
async def search_properties(
    location: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    property_type: Optional[str] = None,
    min_bedrooms: Optional[int] = None,
) -> List[PropertySearchResponse]:
    """Search for properties based on given criteria.
    
    Args:
        location: Area or suburb to search in
        min_price: Minimum price (optional)
        max_price: Maximum price (optional)
        property_type: Type of property (house, apartment, etc.) (optional)
        min_bedrooms: Minimum number of bedrooms (optional)
    """
    search_request = PropertySearchRequest(
        location=location,
        min_price=min_price,
        max_price=max_price,
        property_type=property_type,
        min_bedrooms=min_bedrooms
    )
    return await property_scraper.search_properties(search_request)

@tool
async def analyze_property_images(image_urls: List[str]) -> Dict:
    """Analyze property images to extract features and quality assessment.
    
    Args:
        image_urls: List of URLs pointing to property images
    """
    request = ImageAnalysisRequest(image_urls=image_urls)
    return await image_processor.analyze_property_image(request)

@tool
async def get_property_recommendations(
    properties: List[FirestoreProperty],
    preferences: Dict[str, any],
    limit: int = 5
) -> List[FirestoreProperty]:
    """Get personalized property recommendations based on user preferences.
    
    Args:
        properties: List of properties to analyze
        preferences: User preferences for property matching
        limit: Maximum number of recommendations to return
    """
    user_prefs = UserPreferences(**preferences)
    return await recommender.get_recommendations(properties, user_prefs, limit)

# Augment the LLM with tools
tools = [search_properties, analyze_property_images, get_property_recommendations]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are an intelligent property agent assistant that helps users find and analyze properties. Your goal is to help users find suitable properties through a systematic workflow:

                            1. Initial Property Search:
                               - Verify required search parameters (location is mandatory)
                               - If location missing, ask user for location first
                               - Validate price ranges and convert to proper format (e.g. 1.5M to 1500000)
                               - Use search_properties tool with validated parameters
                            
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
                               
                            Always maintain context between steps and use previous findings to inform next actions. Provide clear explanations for your recommendations and decisions.
                            """
                    )
                ]
                + state["messages"]
            )
        ]
    }

async def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = await tool.ainvoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
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
    from app.models import UserPreference
    search_request = PropertySearchRequest(
        location="NSW-chatswood-2067",
        max_price=3500000,
        property_type="house",
        min_bedrooms=4
    )
    preferences = UserPreferences(
        Style = UserPreference(preference="modern", confidence_score=0.9, weight=0.5),
        Environment = UserPreference(preference="quiet", confidence_score=0.9, weight=0.5),
        Quality= UserPreference(preference="high", confidence_score=0.9, weight=0.5),
    )
    print("🏠 Starting Property Agent Example...")
    
    # Initial property search query
    initial_query = f"""
    recommend properties based on the following search request and preferences:
    search_request: {search_request} 
    preferences: {preferences}
    """
    
    messages = [HumanMessage(content=initial_query)]
    result = await agent.ainvoke({"messages": messages})
    
    print("\n🤖 Agent's Response:")
    for message in result["messages"]:
        print(f"\n{message.content}")
    
    # # Follow-up with preferences for recommendations
    # follow_up_query = """
    # Based on these properties, can you recommend the best options for me?
    # My preferences are:
    # - Modern style
    # - Good natural lighting
    # - Close to public transport
    # - Quiet neighborhood
    # """
    
    # messages.extend(result["messages"])
    # messages.append(HumanMessage(content=follow_up_query))
    
    # result = await agent.ainvoke({"messages": messages})
    
    # print("\n🤖 Agent's Recommendations:")
    # for message in result["messages"]:
    #     print(f"\n{message.content}")

# Run the example if this file is executed directly
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_example())