import os
import sys
from pathlib import Path
import asyncio
import json
import logging
import time
from functools import wraps

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent
sys.path.append(str(backend_dir))

from typing_extensions import Literal
from typing import List, Dict, Optional, Any, Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from langchain_xai import ChatXAI
from langchain_core.tools import tool, ToolException
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
from app.services.sql_service import SQLService
from app.services.duckduckgo_search import DuckDuckGoSearchResults

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
llm = ChatXAI(
    api_key=settings.XAI_API_KEY,
    model="grok-3",
)

response_llm = ChatXAI(
    api_key=settings.XAI_API_KEY,
    model="grok-3",
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

async def process_property(result: PropertySearchResponse, preferences: Optional[str] = None) -> FirestoreProperty:
    # Convert to FirestoreProperty
    import time
    start_time = time.time()
    
    firestore_property = FirestoreProperty.from_search_response(result)
    
    # # Check if property exists in Firestore with analysis
    # firestore_start = time.time()
    # stored_property = await firestore_service.get_property(firestore_property.listing_id)
    # firestore_end = time.time()
    # logger.info(f"Firestore get_property took {firestore_end - firestore_start:.2f}s for {firestore_property.listing_id}")
    
    # # If property exists in Firestore and has analysis and preference is none or empty,  use existing analysis
    # if stored_property and stored_property.analysis:
    #     logger.info(f"Using cached property data for {firestore_property.listing_id}")
    #     return stored_property
    
    # 直接创建任务对象，不使用内部异步函数包装
    tasks = []
    task_times = {}
    task_infos = {}
    
    # Add planning info task if address available
    planning_task = None
    if firestore_property.basic_info.full_address:
        planning_task_start = time.time()
        planning_task = asyncio.create_task(get_planning_info(firestore_property.basic_info.full_address))
        tasks.append(planning_task)
        task_infos['planning_task'] = {
            'start_time': planning_task_start,
            'address': firestore_property.basic_info.full_address
        }
    
    # Add image analysis task if images available
    image_task = None
    if firestore_property.media.image_urls:
        image_task_start = time.time()
        image_task = asyncio.create_task(image_processor.analyze_property_image(
            ImageAnalysisRequest(image_urls=firestore_property.media.image_urls , preferences=preferences)
        ))
        tasks.append(image_task)
        task_infos['image_task'] = {
            'start_time': image_task_start,
            'image_count': len(firestore_property.media.image_urls)
        }
    
    # Add investment metrics task if suburb info available
    if firestore_property.basic_info.suburb and firestore_property.basic_info.postcode:
        investment_start = time.time()
        investment_metrics = investment_service.get_investment_metrics(
            suburb=firestore_property.basic_info.suburb,
            postcode=firestore_property.basic_info.postcode,
            bedrooms=firestore_property.basic_info.bedrooms_count or 2
        )
        investment_end = time.time()
        task_times['investment_metrics'] = investment_end - investment_start
        firestore_property.investment_info = investment_metrics
    try:
        # 运行所有任务前记录时间
        gather_prep_start = time.time()
        logger.info(f"Gather prep start time: {gather_prep_start} for {firestore_property.listing_id}")
        # 运行所有任务并收集结果
        if not tasks:
            logger.info(f"No tasks to run for {firestore_property.listing_id}")
            results = []
        else:
            logger.info(f"Running {len(tasks)} concurrent tasks for {firestore_property.listing_id}")
            gather_start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            gather_end = time.time()
            logger.info(f"All tasks gather took {gather_end - gather_start:.2f}s for {firestore_property.listing_id}")
        
        # 处理规划信息结果
        if planning_task and planning_task in tasks:
            task_index = tasks.index(planning_task)
            planning_result = results[task_index]
            planning_end = time.time()
            planning_duration = planning_end - task_infos['planning_task']['start_time']
            task_times['get_planning_info'] = planning_duration
            
            if isinstance(planning_result, Exception):
                logger.error(f"Error in planning info for {firestore_property.listing_id}: {planning_result}")
            else:
                firestore_property.planning_info = planning_result
                logger.info(f"Planning info completed for {firestore_property.listing_id} in {planning_duration:.2f}s")
        
        # 处理图像分析结果
        if image_task and image_task in tasks:
            task_index = tasks.index(image_task)
            image_result = results[task_index]
            image_end = time.time()
            image_duration = image_end - task_infos['image_task']['start_time']
            task_times['analyze_images'] = image_duration
            
            if isinstance(image_result, Exception):
                logger.error(f"Error in image analysis for {firestore_property.listing_id}: {image_result}")
            else:
                # image_result delete perference field
                if isinstance(image_result, dict):
                    image_result = PropertyAnalysis.model_validate(image_result)
                firestore_property.analysis = image_result
                logger.info(f"Image analysis completed for {firestore_property.listing_id} in {image_duration:.2f}s (analyzed {task_infos['image_task']['image_count']} images)")

        print(firestore_property)          
    except Exception as e:
        logger.error(f"Error in concurrent processing for {firestore_property.listing_id}: {str(e)}")
    
    # 如果需要保存属性数据
    # if not stored_property:
    #     save_start = time.time()
    #     await firestore_service.save_property(firestore_property)
    #     save_end = time.time()
    #     logger.info(f"Final save_property took {save_end - save_start:.2f}s for {firestore_property.listing_id}")
    
    total_time = time.time() - start_time
    logger.info(f"Total process_property took {total_time:.2f}s for {firestore_property.listing_id}")
    logger.info(f"Task times for {firestore_property.listing_id}: {task_times}")
    
    return firestore_property

@tool
async def search_properties(
    location: Annotated[List[str], Field(description="List of suburb locations to search, each location is in format: STATE-SUBURB-POSTCODE")],
    min_price: Annotated[Optional[float], Field(description="Minimum price")] = None,
    max_price: Annotated[Optional[float], Field(description="Maximum price")] = None,
    min_bedrooms: Annotated[Optional[int], Field(description="Minimum number of bedrooms")] = None,
    min_bathrooms: Annotated[Optional[int], Field(description="Minimum number of bathrooms")] = None,
    property_type: Annotated[Optional[List[Literal["house", "apartment", "unit", "townhouse", "villa", "rural"]]], Field(description="List of property types")] = None,
    car_parks: Annotated[Optional[int], Field(description="Number of car parks")] = None,
    land_size_from: Annotated[Optional[float], Field(description="Minimum land size in sqm")] = None,
    land_size_to: Annotated[Optional[float], Field(description="Maximum land size in sqm")] = None,
    preferences: Annotated[Optional[str], Field(description="User preferences for the property")] = None
) -> List[FirestoreProperty]:
    """Fetch listing properties inside a suburb under given constraints.
    
    Returns:
        List of FirestoreProperty objects containing listing properties from the search results. 
    """
    import time
    overall_start = time.time()
    
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
    try:
        search_request = PropertySearchRequest(**search_params)
        
        search_start = time.time()
        results = await property_scraper.search_properties(search_request, max_results=5)
        
        search_end = time.time()
        logger.info(f"Property search took {search_end - search_start:.2f}s, found {len(results)} properties")
        
        # 限制并发处理的批次大小，每批处理2个属性
        batch_size = 10
        all_properties = []
        
        for i in range(0, len(results), batch_size):
            batch_start = time.time()
            batch = results[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(results) + batch_size - 1)//batch_size} with {len(batch)} properties")
            
            batch_props = await asyncio.gather(*[process_property(result, preferences) for result in batch])
            valid_props = [prop for prop in batch_props if prop is not None]
            
            all_properties.extend(valid_props)
            batch_end = time.time()
            logger.info(f"Batch {i//batch_size + 1} processing took {batch_end - batch_start:.2f}s, found {len(valid_props)} valid properties")
        
        overall_end = time.time()
        logger.info(f"Total search_properties processing took {overall_end - overall_start:.2f}s, returning {len(all_properties)} properties")
        
        return all_properties
    except Exception as e:
        logger.error(f"Error in concurrent property processing: {str(e)}")
        overall_end = time.time()
        logger.info(f"Failed search_properties took {overall_end - overall_start:.2f}s")
        return str(e)
    
@tool
async def recommend_from_available_properties(
    location: Annotated[Optional[str], Field(description="location preference")],
    property_type: Annotated[Optional[str], Field(description="property type preference")],
    style: Annotated[Optional[str], Field(description="style preference")],
    features: Annotated[Optional[str], Field(description="features preference")],
    layout: Annotated[Optional[str], Field(description="layout preference")],
    transport: Annotated[Optional[str], Field(description="transport preference")],
    investment: Annotated[Optional[str], Field(description="investment preference")],
    school_district: Annotated[Optional[str], Field(description="school district preference")],
    community: Annotated[Optional[str], Field(description="community preference")]
) -> PropertyRecommendationResponse:
    """Recommend the properties from search_properties tool, based on user preferences.
        
    Returns:
        PropertyRecommendationResponse containing recommended properties
    """
    # This is just a placeholder - actual properties will be injected in tool_node
    return PropertyRecommendationResponse(properties=[])

@tool
async def process_preferences(session_id: str, user_message: Annotated[str, Field(description="user's message to process")]) -> dict:
    """Analyze and store users' preferences for home purchase/investment.
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
    query: Annotated[str, Field(description="comprehensive description of the user's house preferences of the suburb based on the context")],
    filters: Annotated[list[str], Field(description="list of filter conditions for suburb search")] = [],
) -> dict:
    """Search for suburbs based on preferences for houses (e.g., family-friendly, high growth potential, good rental yield) at a broader city or regional level, rather than focusing on specific streets or neighborhoods.

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

# Retry decorator for LLM calls
def retry_llm_invoke(llm_instance, messages, max_retries=3, delay=2):
    """
    Retry function for LLM invocations when errors occur.
    
    Args:
        llm_instance: The LLM instance to call
        messages: Messages to pass to the LLM
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 2)
        
    Returns:
        LLM response or raises the last exception after all retries fail
    """
    last_exception = None
    
    # First attempt
    try:
        response = llm_instance.invoke(messages)
        # Check if response is empty or invalid
        if response is None:
            raise ValueError("LLM returned empty response")
            
        return response
    except Exception as e:
        last_exception = e
        logger.warning(f"Initial LLM invoke failed: {str(e)}")
    
    # Retry attempts if first attempt failed
    for attempt in range(max_retries - 1):
        try:
            # Calculate backoff with jitter
            wait_time = delay * (2 ** attempt) * (0.5 + 0.5 * (time.time() % 1))
            logger.info(f"Retrying in {wait_time:.2f} seconds... (attempt {attempt + 1}/{max_retries - 1})")
            time.sleep(wait_time)
            
            # Retry the call
            logger.info(f"LLM invoke retry attempt {attempt + 1}/{max_retries - 1}")
            response = llm_instance.invoke(messages)
            
            # Check if response is empty or invalid
            if response is None or (isinstance(response, AIMessage) and not response.content):
                raise ValueError("LLM returned empty response")
                
            return response
        except Exception as e:
            last_exception = e
            logger.warning(f"LLM invoke retry attempt {attempt + 1} failed: {str(e)}")
    
    # All retries failed
    logger.error(f"All {max_retries} LLM invoke attempts failed. Last error: {str(last_exception)}")
    raise last_exception

async def llm_call(state: dict) -> AgentMessagesState:
    session_id = state["session_id"]
    # Ensure session exists
    session = await chat_storage.get_session(session_id)
    if not session:
        session = await chat_storage.create_session(session_id)
    
    context = await get_conversation_context(session_id)
    
    try:
        response = retry_llm_invoke(
            llm_with_tools,
            [
                SystemMessage(
content=f"""
You are an AI property agent assistant specialized in recommending suburbs and properties, providing insightful analyses about property markets, suburbs, and related real estate information.

You are assisting a USER interactively to fulfill their property-related tasks. Each time the USER sends a message, we may automatically attach additional context about their current interaction state, such as previously expressed property preferences, viewed property listings, ongoing search criteria, recent conversation history, and more. This contextual information may or may not be directly relevant to the current query; it is up to you to decide.

Your main goal is to follow the USER's instructions at each message.

Work in a Thought → Action → PAUSE → Observation loop.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
At the end of the loop you output an Answer to the user's question.

**IMPORTANT - HANDLING TOOL ERRORS:**
- When you receive an observation that begins with "ERROR:", this indicates a tool execution failure
- You must handle this gracefully without blaming the system or exposing technical details
- Instead, use the error information to:
  1. Understand what went wrong (parameter issues, missing data, etc.)
  2. think about alternative approaches or ask for clarification
  3. Present a helpful, user-friendly explanation
  4. Try a different tool or approach to accomplish the user's goal
- Never respond with technical error details or imply the system is broken
- Frame issues as opportunities for clarification or refinement

Follow these interaction principles:
1. Adaptive Interaction:
   - Quickly identify essential details such as user's budget, preferred locations, investment goals, and property type.
   - Bias towards not asking the user for help if you can find the answer yourself.
   - If the user provides clear preferences, proceed directly to recommendations without excessive clarification.
2. Intelligent Context Awareness:
   - Use contextual information (past preferences, viewed listings, search criteria) intelligently to avoid redundant questions.
   - If a user expresses frustration or impatience, immediately shift towards direct recommendations based on available information.
3. Efficient Tool Usage:
   - process_preferences: Use whenever users express their preferences or accept recommendation from agent, including but not limited to location, price, property type, style, environment, features, quality, layout, transport, investment priorities.
   - search_suburb: Utilize to recommend suitable suburbs based on clearly defined or inferred user preferences. Guide the user conversationally to share their preferences naturally.
   - search_properties: Execute once preferences (especially Suburb location) are sufficiently clear. Clarify briefly if needed, but avoid repetitive questioning.
   - recommend_from_available_properties: Use immediately after property search results are obtained.
   - web_search: solve user's inquiry that can't be perfectly solved by other tools or verify the information from other tools.
4. Natural Language Guidelines:
   - Provide comprehensive yet concise responses, mixing quantitative data (prices, yields, growth statistics) with qualitative insights (lifestyle, amenities).    
   - Proactively suggest adjustments or further refinements only after initial recommendations are given, respecting user autonomy.
   - If ambiguity arises, politely and succinctly ask for clarification before proceeding further.
5. Recommendations:
   - Always back recommendations with clear data points and evidence.
   - Avoid mentioning internal tool processes explicitly to maintain a seamless conversational experience.

previous user's preferences: 
{state["preferences"]}

previous user's requirements:
{state["search_params"]}

properties searched:
{state["available_properties"]}

conversation context:
{context}
"""
                )
            ]
            + state["messages"] # single run message, tools call from latest user message
        )
    except Exception as e:
        logger.error(f"All LLM invoke attempts failed in llm_call: {str(e)}", exc_info=True)
        # Create fallback response that asks the user to try again
        response = AIMessage(content="I'm having trouble processing your request right now. Could you please try again or rephrase your question?")
    
    return AgentMessagesState(
        messages=state["messages"] + [response],
        session_id=session_id,
        preferences=state["preferences"],
        search_params=state["search_params"],
        latest_recommendation=state["latest_recommendation"]
    )

async def safe_tool_call(tool, args, verbose=True):
    """安全地执行工具调用，捕获并处理所有可能的异常
    
    Args:
        tool: 要调用的工具对象
        args: 传递给工具的参数
        verbose: 是否记录详细日志
    
    Returns:
        dict: 包含执行结果或错误信息的字典
    """
    tool_name = getattr(tool, "name", str(tool))
    
    if verbose:
        logger.info(f"Using tool: {tool_name}")
        logger.info(f"Tool arguments: {json.dumps(args, indent=2) if isinstance(args, dict) else args}")
    
    try:
        result = await tool.ainvoke(args)
        return {
            "status": "success",
            "result": result
        }
    except ToolException as e:
        # 工具验证错误（如参数不正确）
        error_msg = f"I couldn't use the {tool_name} tool correctly: {str(e)}. Let me try a different approach."
        logger.warning(f"Tool validation error for {tool_name}: {str(e)}")
        return {
            "status": "error",
            "error_type": "validation",
            "message": error_msg,
            "exception": e
        }
    except Exception as e:
        # 一般执行错误
        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)

        return {
            "status": "error",
            "error_type": "execution",
            "message": f"Error executing tool {tool_name}: {str(e)}",
            "exception": e
        }

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
        
        # 向参数中添加session_id
        if isinstance(args, dict):
            args["session_id"] = session_id
        
        # 使用safe_tool_call安全地执行工具
        response = await safe_tool_call(tool, args)
        
        # 根据执行结果处理
        if response["status"] == "success":
            observation = response["result"]
            
            # 处理特定工具的响应
            if tool.name == "process_preferences" and isinstance(observation, dict):
                # 更新状态中的偏好和搜索参数
                preferences = observation.get("preferences", preferences)
                search_params = observation.get("search_params", search_params)
            
            elif tool.name == "recommend_from_available_properties":
                # 获取推荐历史
                recommendation_history = await chat_storage.get_recommendation_history(session_id)
                # 过滤掉之前推荐的属性
                new_properties = [
                    prop for prop in available_properties 
                    if prop.listing_id not in recommendation_history
                ]
                
                if not new_properties:
                    observation = "No new properties to recommend. need to search properties again."
                else:
                    # 使用推荐服务获取推荐
                    observation = await recommender.get_recommendations(
                        properties=new_properties,
                        preferences=preferences
                    )
                    latest_recommendation = observation
                    logger.info(f"Last recommendation: {latest_recommendation}")
                
                # 保存为聊天消息
                chat_msg = ConversationMessage(
                    role="tool",
                    content="Property recommendations generated.",
                    type="property_recommendation",
                    recommendation=latest_recommendation if isinstance(latest_recommendation, PropertyRecommendationResponse) else None,
                    metadata={"tool_call_id": tool_call["id"], "tool_name": tool.name},
                    timestamp=datetime.now()
                )
                await chat_storage.save_message(session_id, chat_msg)
            
            elif tool.name == "search_properties":
                available_properties = observation
                # 更新聊天存储中的可用属性状态
                await chat_storage.update_recommendation_state(
                    session_id=session_id,
                    available_properties=available_properties
                )
        else:
            # 处理错误情况
            observation = f"ERROR: {response['message']}"
        
        # 添加工具消息到结果中
        result.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            metadata={"type": "property_recommendation"}
        ))

    # 返回更新后的状态
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
    
    try:
        # Extract AI and user messages safely
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        # Log message counts for debugging
        logger.info(f"AI messages count: {len(ai_messages)}")
        logger.info(f"User messages count: {len(user_messages)}")
        
        # Check if we have necessary messages to process
        if not ai_messages or not user_messages:
            logger.warning(f"Missing messages - AI: {len(ai_messages)}, User: {len(user_messages)}")
            # Create a fallback response if no messages are available
            response = AIMessage(content="I'm sorry, but I couldn't process your request properly. Could you please try again?")
        else:
            # Normal processing flow when messages are available
            context = await get_conversation_context(session_id)
            
            # Debug log the messages we're using
            logger.debug(f"Using AI message: {ai_messages[-1].content[:100]}...")
            logger.debug(f"Using user message: {user_messages[-1].content[:100]}...")
            
            # Try to get response from LLM with retry and fallback
            try:
                response = retry_llm_invoke(
                    response_llm,
                    [
                        SystemMessage(content=f"""
You are acting as a guardrail for the final content that will be shown directly to the user.
Your role is to enhance the practicality, fault tolerance, and realistic interactivity of this response before it is delivered.
ONLY output the final response to the user, no other content.
                                      
Your core tasks include:
- A final check on grammar and natural language fluency
- Rewrite any robotic or awkward phrasing
- Detect missing context or user preferences and prompt the user to clarify
- Encourage users to share more personalized needs to improve the quality of recommendations
- Provide concise follow-up suggestions

Optimization Goals:
- Make the interaction feel more natural and conversational — like a professional and thoughtful property consultant.
- Intelligently detect if any important steps or preferences were missed earlier and prompt the user to complete them.
- If preferences were already provided but no matching results are found, proactively offer alternative suggestions or explain why.
- Limit the number of questions to avoid turning the conversation into a rigid form — keep it human and natural.
- Focus on uncovering the user's real intent and priorities, so recommendations can feel tailored and relevant

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
"""),
                    HumanMessage(content=f"Human Question: {user_messages[-1].content}  \n\nCurrent Draft Answer to Review: {ai_messages[-1].content}"),
                    ],
                    max_retries=3
                )
            except Exception as llm_error:
                # All LLM retry attempts failed, use fallback
                logger.error(f"All LLM retry attempts failed in response_node: {str(llm_error)}")
                # Return the existing state without adding a new message
                return AgentMessagesState(
                    messages=state["messages"],
                    session_id=session_id,
                    preferences=state.get("preferences", {}),
                    search_params=state.get("search_params", {}),
                    available_properties=state.get("available_properties", []),
                    latest_recommendation=state.get("latest_recommendation", None)
                )
        
        # Return updated state with response
        return AgentMessagesState(
            messages=state["messages"] + [response],
            session_id=session_id,
            preferences=state.get("preferences", {}),
            search_params=state.get("search_params", {}),
            available_properties=state.get("available_properties", []),
            latest_recommendation=state.get("latest_recommendation", None)
        )
        
    except Exception as e:
        logger.error(f"Error in response_node: {str(e)}", exc_info=True)
        # Create fallback response using last AI message if available
        if ai_messages and len(ai_messages) > 0:
            logger.info("Using last AI message as fallback due to error")
            fallback_response = ai_messages[-1]
        else:
            logger.warning("No AI messages available for fallback, using generic response")
            fallback_response = AIMessage(content="I encountered an error processing your request. Please try again.")
        
        return AgentMessagesState(
            messages=state["messages"] + [fallback_response],
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
# print(search_properties.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(search_suburb.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(process_preferences.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(recommend_from_available_properties.args_schema.schema_json(indent=2))  # Print full schema as JSON
# print(web_search.args_schema.schema_json(indent=2))  # Print full schema as JSON

