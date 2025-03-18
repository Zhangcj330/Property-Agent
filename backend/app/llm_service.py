from typing import Dict, Tuple, List, Annotated, Optional
from pydantic import BaseModel
import operator
import json
from typing_extensions import TypedDict
from enum import Enum

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from app.config import settings
from app.services.property_scraper import PropertyScraper
from app.models import UserPreferences, PropertySearchRequest

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
llm = ChatOpenAI(
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
        
        # Part 2: Smart conversation history management
        # Set a reasonable limit for conversation history (e.g., last 10 exchanges)
        MAX_HISTORY_LENGTH = 10
        recent_history = chat_history[-MAX_HISTORY_LENGTH*2:] if len(chat_history) > MAX_HISTORY_LENGTH*2 else chat_history
        
        # Part 3: Extract explicit preferences from conversation
        extraction_prompt = """You are an expert Australian real estate consultant and demographic analyst. 
        You are specializing in clarifying client requirements for property searches.
        
        Analyze the conversation THOROUGHLY to extract TWO types of information:
        1. Explicit preferences directly stated by the user
        2. Implicit preferences that can be reasonably inferred from context
        
        Rules for making inferences about vague or implied preferences:
        1. For vague terms like "affordable", "nice area", "good schools", make contextual inferences:
           - "Affordable" → Estimate price range based on location mentioned and property type
           - "Nice area" → Infer preferences for safety, amenities, or community characteristics
           - "Good schools" → Mark SchoolDistrict as important without requiring specific school names
        2. Use demographic knowledge to make reasonable assumptions:
           - Families with children ALWAYS need good schools (mark SchoolDistrict as high priority) and outdoor space/garden
           - Young professionals may prioritize commute time and lifestyle amenities
           - Retirees might prefer single-level properties with low maintenance
        3. Assign appropriate IMPORTANCE weights based on how much the user cares about this feature:
           - High weights (0.8-1.0) for explicitly emphasized requirements ("must have", "very important", etc.)
           - Medium weights (0.4-0.7) for standard preferences
           - Lower weights (0.1-0.3) for "nice to have" preferences
        4. SEPARATELY, assign CONFIDENCE scores for each preference:
           - High confidence (0.8-1.0) for explicitly stated preferences
           - Medium confidence (0.4-0.7) for strongly implied preferences
           - Low confidence (0.1-0.3) for demographic inferences or contextual hints
        5. When faced with vague terms, make reasonable DEFAULT ASSUMPTIONS rather than leaving fields empty
        
        IMPORTANT: When a user mentions they have a family with children, ALWAYS include the following inferences:
        - SchoolDistrict: {"preference": "Good schools important for children", "weight": 0.8, "confidence_score": 0.7}  # High importance, medium confidence
        - Features: {"preference": "Child-friendly space, garden or yard for kids", "weight": 0.7, "confidence_score": 0.7}  # Medium importance, medium confidence
        - Community: {"preference": "Family-friendly neighborhood", "weight": 0.7, "confidence_score": 0.7}  # Medium importance, medium confidence
        - Safety: {"preference": "Safe environment for children", "weight": 0.8, "confidence_score": 0.8}  # High importance, high confidence
        
        Handle these common vague scenarios:
        - Budget: If user says "affordable" without specifics, estimate based on location and property type
        - Location: If only city mentioned, include popular/central suburbs as default options
        - Property type: If unclear, infer from lifestyle clues (family→house, single→apartment)
        - Features: Connect lifestyle mentions to likely feature needs
          * Family with children → garden/yard, play areas, storage, separate living zones
          * Professionals → home office, entertainment areas, low maintenance
          * Downsizers → single level, low maintenance, accessibility features

        1. Detailed user preferences with importance weights (0.0-1.0) AND confidence scores (0.0-1.0):
        {
            "Location": {"preference": "specific-location", "weight": importance_weight, "confidence_score": confidence_score},
            "Price": {"preference": "price-range", "weight": importance_weight, "confidence_score": confidence_score},
            "Size": {"preference": "size-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Layout": {"preference": "layout-preferences", "weight": importance_weight, "confidence_score": confidence_score},
            "PropertyType": {"preference": "type-preference", "weight": importance_weight, "confidence_score": confidence_score},
            "Features": {"preference": "desired-features", "weight": importance_weight, "confidence_score": confidence_score},
            "Condition": {"preference": "condition-preference", "weight": importance_weight, "confidence_score": confidence_score},
            "Environment": {"preference": "environment-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Style": {"preference": "style-preference", "weight": importance_weight, "confidence_score": confidence_score},
            "Quality": {"preference": "quality-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Room": {"preference": "room-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "SchoolDistrict": {"preference": "school-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Community": {"preference": "community-preferences", "weight": importance_weight, "confidence_score": confidence_score},
            "Transport": {"preference": "transport-requirements", "weight": importance_weight, "confidence_score": confidence_score},
            "Other": {"preference": "other-preferences", "weight": importance_weight, "confidence_score": confidence_score}
        }

        2. Specific search parameters:
        {
            "location": "STATE-SUBURB-POSTCODE",
            "suburb": "suburb name",
            "state": "state code",
            "postcode": number,
            "min_price": number or null,
            "max_price": number or null,
            "min_bedrooms": number or null,
            "property_type": "house/apartment/etc" or null
        }

        For location format:
        - Use STATE-SUBURB-POSTCODE format (e.g., "NSW-Chatswood-2067")
        - For vague locations, provide state code at minimum, and estimate suburb if possible
        - If only city is mentioned (e.g., "Sydney"), use STATE-City format (e.g., "NSW-Sydney")
        
        For price ranges:
        - When only max price is mentioned, estimate reasonable min_price as 80% of max
        - When only "affordable" is mentioned, estimate based on location and property type
        - For luxury requests, adjust price ranges upward for the specified area
        
        For property types:
        - Map informal terms to standard types (e.g., "unit" → "apartment", "townhouse/villa" → "townhouse")
        - When property type is ambiguous, infer from context (family needs → "house", city lifestyle → "apartment")
        
        Additionally, make INTELLIGENT INFERENCES about unstated preferences:
        - Pay close attention to subtle contextual clues in the conversation
        - Use demographic patterns to fill in knowledge gaps
        - For missing information, apply Australian real estate market knowledge:
          * Sydney suburbs like North Shore, Eastern Suburbs and Inner West are high-priced premium areas
          * Melbourne's inner suburbs (3-12km from CBD) are culturally diverse with good public transport
          * Brisbane offers better affordability compared to Sydney and Melbourne
          * For investors, areas with university proximity often have strong rental demand
          * Coastal and waterfront locations typically command 20-30% premium
        - For vague location mentions, make educated guesses:
          * "Sydney" + "family" + "$1M-1.5M" → Western or Northwest suburbs
          * "Melbourne" + "professional" + "lifestyle" → Inner city or Inner Eastern suburbs
          * "Brisbane" + "affordable" → Southern or Western suburbs
        
        IMPORTANT - UPDATE INSTRUCTIONS:
        1. You must review the user's current preferences summary provided to you.
        2. For conflicting information, consider BOTH importance weight AND confidence score:
           - Preferences with higher importance to the user should generally be preserved
           - New information with high confidence may update preferences with low confidence
           - Explicitly changed preferences should always be updated regardless of importance
        3. When a user clearly changes a preference, update it while maintaining its importance weight
        4. If a user clarifies a previous vague statement, increase the confidence score while preserving importance
        
        You MUST return BOTH objects in the format:
        {
            "user_preferences": {preference object},
            "search_parameters": {search parameters object}
        }
        """
        
        # Build conversation context with emphasis on recent exchanges
        conversation = "\nRecent Conversation:\n"
        for msg in recent_history:
            conversation += f"{msg['role'].title()}: {msg['content']}\n"
        
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

        # Intelligent preference merging with separate importance and confidence handling
        merged_preferences = {**current_preferences}
        for field, value in new_user_prefs.items():
            if field in merged_preferences:
                # 获取当前的值、重要性和置信度
                current_pref = merged_preferences[field]
                current_value = current_pref.get("preference") if current_pref else None
                current_importance = current_pref.get("weight", 0.5) if current_pref else 0.5
                current_confidence = current_pref.get("confidence_score", 0.7) if current_pref else 0.7
                
                # 获取新的值、重要性和置信度
                new_value = value.get("preference") if value else None
                new_importance = value.get("weight", 0.5) if value else 0.5
                new_confidence = value.get("confidence_score", 0.7) if value else 0.7
                
                # 智能合并逻辑:
                # 1. 如果内容变了且新的置信度高于旧的，则更新
                # 2. 保留用户原始的重要性权重(除非新的明显更高)
                # 3. 如果是模糊变为具体，则更新
                should_update = False
                # 根据用户明确的位置和价格变更总是需要更新
                if field in ["Location", "Price"] and current_value != new_value and new_confidence > 0.6:
                    should_update = True
                # 如果内容变了且新的置信度明显更高，则更新
                elif current_value != new_value and new_confidence > current_confidence + 0.2:
                    should_update = True
                # 从模糊变为具体情况
                elif current_value and new_value and "vague" in str(current_value).lower() and "vague" not in str(new_value).lower():
                    should_update = True
                # 内容相同但用户表达了更高的重要性
                elif current_value == new_value and new_importance > current_importance + 0.2:
                    should_update = True
                    
                if should_update:
                    # 在更新时保留较高的重要性权重
                    final_importance = max(current_importance, new_importance)
                    # 创建一个新的UserPreference对象
                    merged_preferences[field] = {
                        "preference": new_value,
                        "weight": final_importance,
                        "confidence_score": new_confidence
                    }
            else:
                # 新偏好字段，直接添加
                merged_preferences[field] = value
                
        # Similar intelligent merging for search parameters
        merged_search_params = {**current_search_params}
        for field, value in new_search_params.items():
            # For search parameters, we prioritize specificity
            if field in merged_search_params:
                current_value = merged_search_params[field]
                
                # For location, favor more specific locations
                if field == "location":
                    # If new value includes suburb or postcode but current doesn't, use new
                    if ("-" in str(value) and "-" not in str(current_value)) or \
                       (str(value).count("-") > str(current_value).count("-")):
                        merged_search_params[field] = value
                # For price, keep the most recent unless it's None
                elif field in ["min_price", "max_price"] and value is not None:
                    merged_search_params[field] = value
                # For other parameters, prefer non-None values
                elif value is not None:
                    merged_search_params[field] = value
            else:
                # New parameter, simply add it if not None
                if value is not None:
                    merged_search_params[field] = value
                
        # Update state with merged values
        state["userpreferences"] = merged_preferences
        state["propertysearchrequest"] = merged_search_params
        return state

    # Worker: Check if all required fields are present
    def check_worker(state: State) -> State:
        """Check if all required preferences are present"""
        # Check if we have enough information for a basic property search
        required_fields = ["location", "max_price", "property_type"]
        required_preferences = ["Style", "SchoolDistrict", "Community"]
        
        # Check each required field
        missing = []
        
        # For search parameters, check if values are present and not None
        for field in required_fields:
            if field not in state["propertysearchrequest"] or state["propertysearchrequest"][field] is None:
                missing.append(field)
        
        # For preferences, check if values exist and are meaningful
        for field in required_preferences:
            # Check if preference exists, has a value, and the value is not None
            if (field not in state["userpreferences"] or 
                not state["userpreferences"][field] or 
                not state["userpreferences"][field].get("preference")):
                missing.append(field)
        
        # Special handling for location - if we have state but not specific suburb
        if "location" in missing and state["propertysearchrequest"].get("state"):
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
            
            For PRICE queries:
            - If user mentions "affordable" or "reasonable", provide specific price ranges for their desired area
            - Example: "For a 3-bedroom house in Melbourne's eastern suburbs, prices typically range from $900K to $1.2M in the current market."
            
            For PROPERTY TYPE queries:
            - If user is unsure, suggest based on their lifestyle needs and budget
            - Example: "Since you mentioned low maintenance and being close to the city, a modern apartment or townhouse might be ideal for your situation."
            
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
        current_preferences = state["userpreferences"]
        current_search_params = state["propertysearchrequest"]
        chat_history = state.get("chat_history", [])
        
        if not chat_history:
            return state
            
        try:
            ambiguity_prompt = """You are an expert Australian Real Estate Advisor specializing in identifying conflicting requirements and ambiguities in client preferences.
            
            Your task is to carefully analyze conversations to detect THREE types of ambiguity:
            1. CONTRADICTIONS: When client states two directly conflicting requirements (e.g., "quiet neighborhood" AND "close to nightlife")
            2. VAGUENESS: When client uses subjective terms without specifics (e.g., "nice area", "affordable", "modern")
            3. UNREALISTIC EXPECTATIONS: When client expectations don't align with market reality (e.g., luxury features at budget prices)
            
            For VAGUE preferences:
            - DO NOT flag common subjective terms like "good schools" or "nice community" as ambiguities unless they're central to the search
            - Only flag vague terms when specificity is ABSOLUTELY NECESSARY for property search
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
                        "field": "location|price|size|property_type|school_district|community",
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
            
            # print(f"Ambiguity check result: has_ambiguities={state['has_ambiguities']}")
            # if state["has_ambiguities"]:
            #     print(f"Found high importance ambiguities: {state['ambiguities']}")
            
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
        self.chat_history = []  # Initialize chat history
        self.property_scraper = PropertyScraper()  # Initialize property scraper
    
    async def process_user_input(self, user_input: str, preferences: Dict = None, search_params: Dict = None) -> Tuple[str, Dict]:
        """Process user input through the graph"""
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Initialize state with chat history
        state = State(
            messages=[HumanMessage(content=user_input)],
            userpreferences=preferences or UserPreferences(),
            propertysearchrequest=search_params or PropertySearchRequest(),
            current_missing_field=None,
            is_complete=False,
            chat_history=self.chat_history
        )
        
        # Run the graph
        final_state = self.graph.invoke(state)
        # print("Final state:", final_state)
        
        # If we have complete preferences, search for properties
        if not final_state["is_complete"]:
            last_message = final_state["messages"][-1].content if final_state["messages"] else ""
            self.chat_history.append({"role": "assistant", "content": last_message})
        
        return final_state

def _importance_to_description(importance: float) -> str:
    """Convert numerical importance to descriptive text"""
    if importance >= 0.9:
        return "Critical"
    elif importance >= 0.7:
        return "High"
    elif importance >= 0.4:
        return "Medium"
    else:
        return "Low"

def _confidence_to_description(confidence: float) -> str:
    """Convert numerical confidence to descriptive text"""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"