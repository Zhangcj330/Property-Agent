from typing import Dict, Tuple, List, Annotated, Optional
from pydantic import BaseModel
import operator
import json
from typing_extensions import TypedDict
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from app.config import settings
from app.services.property_scraper import PropertyScraper
from app.models import UserPreferences, PropertySearchRequest, ChatMessage
from app.services.chat_storage import ChatStorageService

# Graph state
class State(TypedDict):
    messages: MessagesState
    userpreferences: Dict
    propertysearchrequest: Dict
    current_missing_field: str
    is_complete: bool
    chat_history: List[Dict]  # Add chat history to state
    has_ambiguities: Optional[bool]
    ambiguities: Optional[List[Dict]]


class UserPreferencesSearch(BaseModel):
    user_preferences: UserPreferences
    search_parameters: PropertySearchRequest

# Default LLM
llm = ChatGoogleGenerativeAI(
    api_key=settings.GEMINI_API_KEY,
    base_url=settings.BASE_URL,
    model="gemini-2.0-flash"
)



# Structured output parser
def PreferenceGraph():
    # Worker: Extract preferences
    def extract_worker(state: State) -> State:
        """Extract search parameters and preferences from user input"""
        if "chat_history" not in state or len(state["chat_history"]) == 0:
            return state
            
        chat_history = state["chat_history"]
        current_preferences = state["userpreferences"]
        current_search_params = state["propertysearchrequest"]
        current_message = state["messages"][0].content if state["messages"] else None  # 获取当前用户输入
        
        print(f"Starting extract_worker with current_search_params: {current_search_params}")
        
        # Build preference summary for context
        preference_summary = ""
        if current_preferences:
            preference_summary += "Current preferences:\n"
            for field, pref in current_preferences.items():
                # Use the new UserPreference structure
                preference_value = pref.get("preference")
                importance = pref.get("weight")
                confidence = pref.get("confidence_score")
                
                if preference_value is not None:
                    importance_desc = _importance_to_description(importance)
                    confidence_desc = _confidence_to_description(confidence)
                    preference_summary += f"- {field.capitalize()}: {preference_value} (Importance: {importance_desc}, Confidence: {confidence_desc})\n"
        
        if current_search_params:
            preference_summary += "\nCurrent search parameters:\n"
            for field, value in current_search_params.items():
                if value is not None:
                    preference_summary += f"- {field.capitalize()}: {value}\n"
        
        # Smart conversation history management
        MAX_HISTORY_LENGTH = 10
        recent_history = chat_history[-MAX_HISTORY_LENGTH*2:] if len(chat_history) > MAX_HISTORY_LENGTH*2 else chat_history
        
        # Extract explicit preferences from conversation
        extraction_prompt = """You are an expert Australian real estate consultant and demographic analyst. 
        You are specializing in clarifying client requirements for property searches.
        
        Analyze the conversation THOROUGHLY to extract TWO types of information:
        1. Explicit preferences directly stated by the user
        2. Implicit preferences that can be reasonably inferred from context with high confidence
        
        IMPORTANT: Only return preferences and search parameters that are CLEARLY stated or can be CONFIDENTLY inferred.
        If you cannot determine a preference with reasonable confidence, DO NOT include it in your response.
        
        Rules for making inferences about preferences:
        1. For vague terms, only make inferences when context provides strong support:
           - "Affordable" → Only estimate price range if location and property type are known
           - "Nice area" → Only infer specific characteristics if contextual clues exist
        2. For demographic inferences, only include when strongly supported by conversation:
           - Only infer family needs if children are explicitly mentioned
           - Only infer commute preferences if work location is mentioned
        3. Assign appropriate IMPORTANCE weights based on user emphasis:
           - High weights (0.8-1.0) for explicitly emphasized requirements ("must have", "very important")
           - Medium weights (0.4-0.7) for standard preferences
           - Lower weights (0.1-0.3) for "nice to have" preferences
        4. Assign CONFIDENCE scores honestly reflecting certainty:
           - High confidence (0.8-1.0) only for explicitly stated preferences
           - Medium confidence (0.4-0.7) for strongly implied preferences
           - Low confidence (0.1-0.3) for reasonable inferences with some support
        5. When faced with uncertainty, OMIT the field rather than making weak assumptions
        
        1. Only include user preferences that are clear from the conversation:
        {
            "Layout": {"preference": "layout-preferences", "weight": importance_weight, "confidence_score": confidence_score},
            "Features": {"preference": "desired-features", "weight": importance_weight, "confidence_score": confidence_score},
            "Condition": {"preference": "condition-preference", "weight": importance_weight, "confidence_score": confidence_score},
            "Environment": {"preference": "environment-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Style": {"preference": "style-preference", "weight": importance_weight, "confidence_score": confidence_score},
            "Quality": {"preference": "quality-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "SchoolDistrict": {"preference": "school-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Community": {"preference": "community-preferences", "weight": importance_weight, "confidence_score": confidence_score},
            "Transport": {"preference": "transport-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Other": {"preference": "other-preferences", "weight": importance_weight, "confidence_score": confidence_score}
        }

        2. Only include search parameters that can be confidently determined:
        {
            "location": ["STATE-SUBURB-POSTCODE"], // List of locations in STATE-SUBURB-POSTCODE format
            "min_price": number or null,
            "max_price": number or null,
            "min_bedrooms": number or null,
            "min_bathrooms": number or null,
            "property_type": ["House", "Unit", "Apartment", "Studio", "Townhouse", "Land", "Villa", "Rural"], // List of property types
            "car_parks": number or null,
            "land_size_from": number or null, // Minimum land size in sqm
            "land_size_to": number or null, // Maximum land size in sqm
            "geo_location": [latitude, longitude] or null // Array of two floating point numbers
        }

        For location format:
        - Each location should be in STATE-SUBURB-POSTCODE format (e.g., "NSW-Chatswood-2067")
        - Only include specific suburbs if clearly mentioned or strongly implied
        - Return locations as an ARRAY of strings, even if there's only one location
        - ALWAYS include the STATE code (NSW, VIC, QLD, etc.) at the beginning of each location
        
        For property types:
        - Only include property types explicitly mentioned or strongly implied
        - Return property types as an ARRAY of strings
        
        For price ranges:
        - Only include price information that is explicitly stated or can be confidently inferred
        - Omit price fields if there is significant uncertainty
        
        IMPORTANT - UPDATE INSTRUCTIONS:
        1. Review the user's current preferences summary provided to you.
        2. ONLY return fields that need to be updated or added based on new information.
        3. For each field, determine if the new information justifies an update:
           - Clear changes in user preference always justify updates
           - Higher confidence information should replace lower confidence information
           - More specific information should replace vague information
        4. Omit any fields where no update is needed or where uncertainty remains high
        
        You MUST return BOTH objects in the format, including ONLY fields that need updating:
        {
            "user_preferences": {only preferences that need updating},
            "search_parameters": {only search parameters that need updating}
        }
        """

        # Build conversation context with emphasis on recent exchanges
        conversation = "\nRecent Conversation:\n"
        for msg in recent_history:
            conversation += f"{msg['role'].title()}: {msg['content']}\n"
        
        # 添加当前用户输入
        if current_message:
            conversation += f"User: {current_message}\n"
        
        # Add the preference summary to the context
        context_with_memory = f"{preference_summary}\n\n{conversation}"
        
        # Get LLM response
        response = llm.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=context_with_memory)
        ])
        
        # Parse extracted preferences
        output = JsonOutputParser(pydantic_object=UserPreferencesSearch).invoke(response)
        
        # Extract the relevant portions and parse them
        new_search_params = output["search_parameters"]
        new_user_prefs = output["user_preferences"]

        # Simplified preference merging logic
        # Only update fields that were returned by the LLM
        merged_preferences = {**current_preferences}
        for field, value in new_user_prefs.items():
            if field in merged_preferences:
                # Get the current preference
                current_pref = merged_preferences[field]
                current_confidence = current_pref.get("confidence_score", 0.5) if current_pref else 0.5
                
                # Get the new preference confidence
                new_confidence = value.get("confidence_score", 0.5) if value else 0.5
                
                # Update if new confidence is higher or equal
                if new_confidence >= current_confidence:
                    # Preserve the higher weight if available
                    if current_pref and "weight" in current_pref and "weight" in value:
                        value["weight"] = max(current_pref["weight"], value["weight"])
                    merged_preferences[field] = value
            else:
                # Add new preference
                merged_preferences[field] = value
                
        # Simplified search parameters merging
        merged_search_params = {**current_search_params}
        if new_search_params:
            for field, value in new_search_params.items():
                merged_search_params[field] = value

        # Update state with merged values
        state["userpreferences"] = merged_preferences
        state["propertysearchrequest"] = merged_search_params
        
        # Debug information
        print(f"Updated search params: {merged_search_params}")
        
        return state

    # Worker: Check if all required fields are present
    def check_worker(state: State) -> State:
        """Check if all required preferences are present"""
        # Check if we have enough information for a basic property search
        required_fields = ["location", "max_price", "property_type"]
        required_preferences = ["Style", "SchoolDistrict", "Community"]
        
        # Check each required field
        missing = []
        
        # For search parameters, check if values are present and not None or empty lists
        for field in required_fields:
            if field not in state["propertysearchrequest"] or state["propertysearchrequest"][field] is None:
                missing.append(field)
            elif field in ["location", "property_type"]:
                # These fields should now be lists
                value = state["propertysearchrequest"][field]
                if not isinstance(value, list) or len(value) == 0:
                    missing.append(field)
        
        # For preferences, check if values exist and are meaningful
        for field in required_preferences:
            # Check if preference exists, has a value, and the value is not None or "Not specified"
            if (field not in state["userpreferences"] or 
                not state["userpreferences"][field] or 
                not state["userpreferences"][field].get("preference") or
                state["userpreferences"][field].get("preference") == "Not specified"):
                missing.append(field)
        
        # Special handling for location - if we have state but not specific suburb
        if "location" in missing and "location" in state["propertysearchrequest"]:
            locations = state["propertysearchrequest"]["location"]
            if isinstance(locations, list) and len(locations) > 0:
                # Check if we have at least state information
                has_state_info = any("-" in loc for loc in locations)
                if has_state_info:
                    # If we at least have the state, consider location as partially filled
                    # and prioritize other missing fields first
                    missing.remove("location")
                    if not missing:  # If no other fields are missing, we can consider location
                        missing.append("location")
        
        if missing:
            # Set the first missing field as current field
            state["current_missing_field"] = missing[0]
            state["has_missing_fields"] = True
            # print(f"Missing fields: {missing}")
        else:
            state["is_complete"] = True
            # print("All required fields collected")
            
        return state

    # Worker: Generate follow-up question
    def question_worker(state: State) -> State:
        """Generate follow-up question for missing information"""
        current_missing_field = state["current_missing_field"]
        search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        # Build conversation context
        conversation = "\nPrevious conversation:\n"
        for msg in chat_history:
            conversation += f"{msg['role'].title()}: {msg['content']}\n"
        
        # Check how many exchanges have occurred about this particular field
        field_exchanges = 0
        for i in range(len(chat_history) - 1, -1, -2):  # Start from most recent assistant message
            if i >= 1 and current_missing_field.lower() in chat_history[i]["content"].lower():
                field_exchanges += 1
            else:
                break
        
        # Generate natural follow-up question
        response = llm.invoke([
            SystemMessage(content="""You are a knowledgeable and empathetic Australian real estate agent with years of experience.
            Your goal is to help users find their ideal property by gathering information and providing expert guidance.
            
            Guidelines for asking follow-up questions:
            - Be conversational and empathetic
            - Ask one clear question at a time
            - If the user seems undecided or has provided vague answers after 1-2 exchanges:
                * Provide specific market insights or suggestions based on their context
                * Share relevant examples with actual suburb names and price points
                * Explain trade-offs between different options concretely
                * Make professional recommendations based on their needs
            
            For LOCATION queries:
            - If user is vague about location, suggest specific suburbs based on their other requirements
            - Example: "Based on your budget and preference for good schools, suburbs like Chatswood, Epping, or Eastwood in Sydney could be good options."
            - Remember that multiple locations can be searched, so ask if they'd like to include multiple areas
            
            For PRICE queries:
            - If user mentions "affordable" or "reasonable", provide specific price ranges for their desired area
            - Example: "For a 3-bedroom house in Melbourne's eastern suburbs, prices typically range from $900K to $1.2M in the current market."
            
            For PROPERTY TYPE queries:
            - If user is unsure, suggest based on their lifestyle needs and budget
            - Example: "Since you mentioned low maintenance and being close to the city, a modern apartment or townhouse might be ideal for your situation."
            - Remember that multiple property types can be selected, so ask if they'd like to include multiple types
            
            For STYLE queries:
            - Offer examples of common styles with brief descriptions
            - Example: "In terms of style, are you leaning more towards modern minimalist, traditional character, or perhaps something in between like contemporary with classic elements?"
            
            When the user has been asked about the same topic multiple times (3+):
            - STOP asking repetitive questions
            - Make a reasonable assumption and move on
            - Example: "I understand you're not completely decided on the exact location. Let's assume we'll focus on Sydney's northern suburbs for now, and we can refine this later if needed."
            
            Current missing field: {current_missing_field}
            Current search parameters: {search_params}
            Field exchanges so far: {field_exchanges}
            """),
            HumanMessage(content=f"""Based on the conversation:
            {conversation}
            
            We need information about: {current_missing_field}
            Current search parameters: {search_params}
            
            This is exchange #{field_exchanges + 1} about this field.
            
            Ask a natural follow-up question to get the missing information. If field_exchanges >= 2, the user seems undecided or vague about this topic, so provide specific examples and suggestions rather than asking another general question.
            """)
        ])
        
        # Add question to messages
        state["messages"].append(AIMessage(content=response.content))
        
        # Add to chat history
        if "chat_history" in state:
            state["chat_history"].append({"role": "assistant", "content": response.content})
            
        return state

    # Worker: Check for ambiguities in user preferences
    def ambiguity_worker(state: State) -> State:
        """Identify and resolve ambiguities in user preferences and search parameters"""
        chat_history = state.get("chat_history", [])
        current_message = state["messages"][0].content if state["messages"] else None
        
        if not chat_history and not current_message:
            return state
        
        try:
            ambiguity_prompt = """You are an expert Australian Real Estate Advisor specializing in identifying conflicting requirements and ambiguities in client preferences.
            give subjective recommendations and suggestions for the user based on your knowledge of the market when you find users fail to clarify their preferences more clearly.

            Your task is to carefully analyze conversations to detect THREE types of ambiguity:
            1. CONTRADICTIONS: When client states two directly conflicting requirements (e.g., "quiet neighborhood" AND "close to nightlife")
            2. VAGUENESS: When client uses subjective terms without specifics (e.g., "nice area", "affordable", "modern")
            3. UNREALISTIC EXPECTATIONS: When client expectations don't align with market reality (e.g., luxury features at budget prices)
            
            For VAGUE preferences:
            - DO NOT flag common subjective terms like "good schools" or "nice community" as ambiguities unless they're central to the search
            - Only flag vague terms when specificity is ABSOLUTELY NECESSARY for property search, for example, when the user mentions a broad area like "Sydney" or "Melbourne" without specifying a suburb.
            - For moderately vague terms, make reasonable inferences rather than flagging as ambiguities
            
            For CONTRADICTORY preferences:
            - Minor contradictions (e.g., "prefer A but B is also okay") are NOT true contradictions 
            - Only flag as contradictions when two requirements are truly incompatible
            - Assess whether the contradiction will significantly impact search results
            
            For UNREALISTIC expectations:
            - Only flag when the gap between expectations and market reality is substantial
            - Consider both price and feature misalignment
            
            Importance rating criteria:
            - "high": Critical issue that MUST be resolved before property search
            - "medium": Notable issue that would benefit from clarification
            - "low": Minor issue that can be resolved through reasonable assumptions
            
            When writing clarification questions:
            1. Acknowledge both contradictory preferences with empathy
            2. Explain the trade-off or market reality briefly
            3. Ask for prioritization rather than elimination
            4. Offer examples or alternatives when appropriate
            
            Example clarification question for contradictions:
            "You mentioned wanting both a quiet neighborhood and easy access to nightlife. These can sometimes be at odds. Which would you prioritize more - the peaceful setting or the proximity to entertainment options? Or perhaps there's a specific balance you're looking for?"
            
            If no significant ambiguities are found (or they can be resolved through reasonable inference), return:
            {"ambiguities": [], "has_ambiguities": false}

            When ambiguities exist, return AT MOST the 2 most critical issues in JSON format:          
            {
                "ambiguities": [
                    {
                        "type": "contradiction|vagueness|unrealistic",
                        "field": "location|price|size|property_type",
                        "description": "Description of the ambiguity",
                        "importance": "high|medium|low",
                        "clarification_question": "Suggested question to resolve ambiguity"
                    }
                ],
                "has_ambiguities": true|false
            }
            """
            
            # Build conversation context
            conversation = "\nConversation context:\n"
            for msg in chat_history:
                conversation += f"{msg['role'].title()}: {msg['content']}\n"
            
            # 添加当前用户输入
            if current_message:
                conversation += f"User: {current_message}\n"
            
            # Get ambiguity analysis from LLM
            response = llm.invoke([
                SystemMessage(content=ambiguity_prompt),
                HumanMessage(content=f"""Based on this conversation:
                {conversation}
                
                Identify any ambiguities or contradictions in the user's requirements:""")
            ])
            
            # Parse ambiguity analysis
            parser = JsonOutputParser()
            analysis = parser.invoke(response)
            
            # Filter ambiguities to only include high importance ones
            high_importance_ambiguities = [
                ambiguity for ambiguity in analysis.get("ambiguities", [])
                if ambiguity.get("importance") == "high"
            ]
            
            # Update has_ambiguities flag based on filtered list
            has_high_importance_ambiguities = len(high_importance_ambiguities) > 0
            
            # Store ambiguity information in state
            state["has_ambiguities"] = has_high_importance_ambiguities
            state["ambiguities"] = high_importance_ambiguities
            
        except Exception as e:
            print(f"Error in ambiguity_worker: {e}")
            state["has_ambiguities"] = False
            state["ambiguities"] = []
        
        return state

    # Worker: Generate ambiguity clarification question
    def ambiguity_question_worker(state: State) -> State:
        """Generate follow-up question to clarify ambiguities"""
        ambiguities = state.get("ambiguities", [])
        
        if not ambiguities:
            return state
            
        # Get the most important ambiguity
        ambiguity = ambiguities[0]
        clarification_question = ambiguity["clarification_question"]
        
        # Add clarification question to messages
        state["messages"].append(AIMessage(content=clarification_question))
        
        # Add to chat history
        if "chat_history" in state:
            state["chat_history"].append({"role": "assistant", "content": clarification_question})
        
        # print(f"Asking for clarification: {clarification_question}")
        
        return state
    
    def _importance_to_description(importance: str) -> str:
        if importance == "high":
            return "Critical issue that MUST be resolved before property search"
        elif importance == "medium":
            return "Notable issue that would benefit from clarification"
        else:
            return "Minor issue that can be resolved through reasonable assumptions"
        
    def _confidence_to_description(confidence: str) -> str:
        if confidence == "high":
            return "High confidence in the information"
        elif confidence == "medium":
            return "Medium confidence in the information"
        else:
            return "Low confidence in the information"
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("extract", extract_worker)
    workflow.add_node("ambiguity", ambiguity_worker)
    workflow.add_node("ambiguity_question", ambiguity_question_worker)
    workflow.add_node("check", check_worker)
    workflow.add_node("question", question_worker)
    
    # Update edges
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "ambiguity")
    
    # Add conditional edge from ambiguity worker
    workflow.add_conditional_edges(
        "ambiguity",
        lambda x: "ambiguity_question" if x.get("has_ambiguities", False) else "check"
    )
    
    # Connect ambiguity question to END
    workflow.add_edge("ambiguity_question", END)
    
    # Connect check to either END or question
    workflow.add_conditional_edges(
        "check",
        lambda x: END if x["is_complete"] else "question"
    )
    workflow.add_edge("question", END)
    
    return workflow.compile()

class LLMService:
    def __init__(self):
        self.graph = PreferenceGraph()
        self.chat_storage = ChatStorageService()
    
    async def process_user_input(
        self, 
        session_id: str, 
        user_input: str, 
        preferences: Dict = None, 
        search_params: Dict = None
    ) -> Dict:
        """处理用户输入，维护对话历史"""
        # 获取或创建会话
        session = await self.chat_storage.get_session(session_id)
        if not session:
            session = await self.chat_storage.create_session(session_id)  # 传入 session_id
            
        # 保存用户消息
        user_message = ChatMessage(
            role="user",
            content=user_input
        )
        await self.chat_storage.save_message(session.session_id, user_message)
        
        # 准备状态
        state = State(
            messages=[HumanMessage(content=user_input)],
            userpreferences=preferences or session.preferences or {},
            propertysearchrequest=search_params or session.search_params or {},
            current_missing_field=None,
            is_complete=False,
            chat_history=[msg.model_dump() for msg in session.messages]  # 使用完整的会话历史
        )
        
        # 运行对话图
        final_state = self.graph.invoke(state)
        
        # 如果有新的回复，保存助手消息
        if final_state["messages"]:
            assistant_message = ChatMessage(
                role="assistant",
                content=final_state["messages"][-1].content
            )
            await self.chat_storage.save_message(session.session_id, assistant_message)
        
        # 更新会话状态
        await self.chat_storage.update_session_state(
            session_id=session.session_id,
            preferences=final_state["userpreferences"],
            search_params=final_state["propertysearchrequest"]
        )
        
        return final_state
